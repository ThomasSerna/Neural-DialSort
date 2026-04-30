from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def read_initializer_scalar(model: onnx.ModelProto, name: str, default: int | None = None) -> int:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            arr = onnx.numpy_helper.to_array(initializer)
            return int(arr.reshape(-1)[0])

    if default is None:
        raise ValueError(
            f"Initializer '{name}' was not found in the model. "
            f"Pass the value manually with --min-value."
        )

    return int(default)


def infer_histogram_size(model: onnx.ModelProto) -> int | None:
    if not model.graph.output:
        return None

    output_type = model.graph.output[0].type
    if not output_type.HasField("tensor_type"):
        return None

    shape = output_type.tensor_type.shape
    if len(shape.dim) != 1:
        return None

    dim = shape.dim[0]
    if dim.HasField("dim_value"):
        return int(dim.dim_value)

    return None


def parse_array(text: str) -> np.ndarray:
    text = text.strip()

    if not text:
        raise ValueError("The array cannot be empty.")

    if text.startswith("["):
        data: Any = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list, for example: [3, 1, 2].")
        return np.asarray(data, dtype=np.int64)

    parts = [p for p in re.split(r"[,\s;]+", text) if p]
    return np.asarray([int(p) for p in parts], dtype=np.int64)


def project_histogram(histogram: np.ndarray, min_value: int) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.int64)

    if histogram.ndim != 1:
        raise ValueError(f"Histogram must be 1D, but got shape {histogram.shape}.")

    if np.any(histogram < 0):
        raise ValueError("Histogram cannot contain negative counts.")

    values = np.arange(min_value, min_value + histogram.shape[0], dtype=np.int64)
    return np.repeat(values, histogram)


def is_non_decreasing(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.all(values[:-1] <= values[1:]))


def get_npu_provider_options() -> dict[str, str]:
    """
    Provider options básicos para Ryzen AI.

    Para Ryzen AI 300 / Strix / Krackan normalmente basta {"log_level": "info"}.
    Para Phoenix/Hawk Point, AMD pide opciones extra con xclbin.
    Puedes forzar ese modo con:
        set RYZEN_AI_NPU_TYPE=PHX_HPT
    """
    npu_type = os.environ.get("RYZEN_AI_NPU_TYPE", "").strip().upper()

    if npu_type not in {"PHX_HPT", "PHX/HPT"}:
        return {"log_level": "info"}

    install_dir = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if not install_dir:
        raise RuntimeError(
            "RYZEN_AI_INSTALLATION_PATH no está definida. "
            "Activa el entorno conda de Ryzen AI o define esa variable."
        )

    xclbin_file = Path(install_dir) / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"

    return {
        "target": "X1",
        "xlnx_enable_py3_round": "0",
        "xclbin": str(xclbin_file),
        "log_level": "info",
    }


def create_session(model_path: str | Path, provider: str) -> ort.InferenceSession:
    print(f"Available providers: {ort.get_available_providers()}")

    session_options = ort.SessionOptions()

    if provider in {"npu", "npu_cpu"}:
        session_options.log_severity_level = 0  # verbose
    else:
        session_options.log_severity_level = 2  # warning

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    elif provider == "npu":
        providers = ["VitisAIExecutionProvider"]
        provider_options = [get_npu_provider_options()]
    elif provider == "npu_cpu":
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
        provider_options = [get_npu_provider_options(), {}]
    else:
        raise ValueError("provider must be one of: cpu, npu, npu_cpu")

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
        provider_options=provider_options,
    )


def run_model(model_path: str | Path, keys: np.ndarray, provider: str) -> tuple[np.ndarray, str, str, list[str]]:
    session = create_session(model_path, provider)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    histogram = session.run([output_name], {input_name: keys.astype(np.int64)})[0]
    histogram = np.asarray(histogram, dtype=np.int64)

    return histogram, input_name, output_name, session.get_providers()


def validate_input_range(keys: np.ndarray, min_value: int, universe_size: int) -> None:
    if keys.ndim != 1:
        raise ValueError(f"Input array must be 1D, but got shape {keys.shape}.")

    if keys.size == 0:
        raise ValueError("Input array cannot be empty.")

    low = min_value
    high_exclusive = min_value + universe_size

    invalid_mask = (keys < low) | (keys >= high_exclusive)
    if np.any(invalid_mask):
        invalid_values = keys[invalid_mask]
        sample = invalid_values[:10].tolist()
        raise ValueError(
            "Some values are outside the model universe. "
            f"Valid range: [{low}, {high_exclusive - 1}]. "
            f"Invalid values detected: {sample}"
        )


def build_keys_from_args(args: argparse.Namespace, *, min_value: int, universe_size: int) -> np.ndarray:
    if args.array is not None:
        return parse_array(args.array)

    if args.random_n is not None:
        rng = np.random.default_rng(args.seed)
        return rng.integers(
            low=min_value,
            high=min_value + universe_size,
            size=args.random_n,
            dtype=np.int64,
        )

    default = np.asarray([3, 1, 3, 5, 1, 3, 5, 3], dtype=np.int64)

    if np.any(default < min_value) or np.any(default >= min_value + universe_size):
        rng = np.random.default_rng(args.seed)
        default = rng.integers(
            low=min_value,
            high=min_value + universe_size,
            size=min(16, max(1, universe_size)),
            dtype=np.int64,
        )

    return default


def compact_array(values: np.ndarray, limit: int = 64) -> str:
    values = np.asarray(values)
    if values.size <= limit:
        return str(values.tolist())
    head = values[:limit].tolist()
    return f"{head} ... total={values.size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a DialSort ONNX model, run CPU/NPU provider, project the sorted vector, and verify correctness."
    )

    parser.add_argument("--model", required=True, help="Path to the .onnx model.")
    parser.add_argument("--array", type=str, default=None, help='Manual array. Example: "3,1,3,5".')
    parser.add_argument("--random-n", type=int, default=None, help="Generate a random array of length N.")
    parser.add_argument("--seed", type=int, default=20260321, help="Seed for --random-n.")
    parser.add_argument("--min-value", type=int, default=None, help="Minimum universe value.")
    parser.add_argument("--u", type=int, default=None, help="Universe size.")
    parser.add_argument("--show-full", action="store_true", help="Print complete arrays.")
    parser.add_argument(
        "--provider",
        choices=["cpu", "npu", "npu_cpu"],
        default="cpu",
        help="Execution provider: cpu, npu, or npu_cpu fallback.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    model = onnx.load(model_path)

    min_value = (
        int(args.min_value)
        if args.min_value is not None
        else read_initializer_scalar(model, "min_value")
    )

    universe_size = int(args.u) if args.u is not None else infer_histogram_size(model)
    if universe_size is None:
        raise ValueError("Could not infer U from the model output. Pass it manually with --u.")

    keys = build_keys_from_args(args, min_value=min_value, universe_size=universe_size)
    validate_input_range(keys, min_value=min_value, universe_size=universe_size)

    histogram, input_name, output_name, active_providers = run_model(model_path, keys, args.provider)
    sorted_from_histogram = project_histogram(histogram, min_value=min_value)

    expected_histogram = np.bincount(keys - min_value, minlength=universe_size).astype(np.int64)
    expected_sorted = np.sort(keys)

    histogram_ok = np.array_equal(histogram, expected_histogram)
    sorted_ok = np.array_equal(sorted_from_histogram, expected_sorted)
    ordered_ok = is_non_decreasing(sorted_from_histogram)
    same_length_ok = sorted_from_histogram.size == keys.size

    all_ok = histogram_ok and sorted_ok and ordered_ok and same_length_ok

    print("\n=== DialSort ONNX Tester ===")
    print(f"Model          : {model_path}")
    print(f"Requested EP   : {args.provider}")
    print(f"Session EPs    : {active_providers}")
    print(f"Input ONNX     : {input_name}")
    print(f"Output ONNX    : {output_name}")
    print(f"N              : {keys.size}")
    print(f"U              : {universe_size}")
    print(f"min_value      : {min_value}")
    print(f"Valid range    : [{min_value}, {min_value + universe_size - 1}]")

    limit = keys.size if args.show_full else 64

    print("\n--- Data ---")
    print(f"Unsorted input             : {compact_array(keys, limit=limit)}")
    print(f"ONNX histogram             : {compact_array(histogram, limit=limit)}")
    print(f"Projected sorted vector    : {compact_array(sorted_from_histogram, limit=limit)}")

    print("\n--- Validation ---")
    print(f"ONNX histogram == np.bincount   : {histogram_ok}")
    print(f"H -> vector projection correct : {sorted_ok}")
    print(f"Vector is sorted               : {ordered_ok}")
    print(f"Original length is preserved   : {same_length_ok}")

    if not all_ok:
        print("\nResult: ERROR")
        raise SystemExit(1)

    print("\nResult: OK")


if __name__ == "__main__":
    main()
