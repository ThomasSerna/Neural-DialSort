from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def read_initializer_scalar(model: onnx.ModelProto, name: str, default: int | None) -> int:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            values = onnx.numpy_helper.to_array(initializer).reshape(-1)
            return int(values[0])

    if default is None:
        raise ValueError(f"Initializer '{name}' was not found. Pass it with --min-value.")

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
        raise ValueError("The input array cannot be empty.")

    if text.startswith("["):
        data: Any = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list, for example [3, 1, 2].")
        return np.asarray(data, dtype=np.int64)

    parts = [part for part in re.split(r"[,\s;]+", text) if part]
    return np.asarray([int(part) for part in parts], dtype=np.int64)


def project_histogram(histogram: np.ndarray, min_value: int) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.int64)
    if histogram.ndim != 1:
        raise ValueError(f"Histogram must be 1D, got shape {histogram.shape}.")
    if np.any(histogram < 0):
        raise ValueError("Histogram contains negative counts.")

    values = np.arange(min_value, min_value + histogram.shape[0], dtype=np.int64)
    return np.repeat(values, histogram)


def build_keys(args: argparse.Namespace, *, min_value: int, universe_size: int) -> np.ndarray:
    if args.array is not None:
        keys = parse_array(args.array)
    elif args.random_n is not None:
        rng = np.random.default_rng(args.seed)
        keys = rng.integers(
            low=min_value,
            high=min_value + universe_size,
            size=args.random_n,
            dtype=np.int64,
        )
    else:
        keys = np.asarray([3, 1, 3, 5, 1, 3, 5, 3], dtype=np.int64)

    low = min_value
    high = min_value + universe_size
    invalid = (keys < low) | (keys >= high)
    if np.any(invalid):
        sample = keys[invalid][:10].tolist()
        raise ValueError(
            f"Input has values outside [{low}, {high - 1}]. Invalid sample: {sample}"
        )

    return keys


def compact(values: np.ndarray, limit: int) -> str:
    if values.size <= limit:
        return str(values.tolist())
    return f"{values[:limit].tolist()} ... total={values.size}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ONNX-Accelerated-DialSort model and verify its histogram output."
    )
    parser.add_argument("--model", required=True, help="Path to a generated .onnx model")
    parser.add_argument("--array", default=None, help='Manual array, e.g. "3,1,3,5"')
    parser.add_argument("--random-n", type=int, default=None, help="Generate random input")
    parser.add_argument("--seed", type=int, default=20260321, help="Random seed")
    parser.add_argument("--min-value", type=int, default=None, help="Minimum key value")
    parser.add_argument("--u", type=int, default=None, help="Universe size")
    parser.add_argument("--show-full", action="store_true", help="Print full arrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    model = onnx.load(model_path)
    min_value = read_initializer_scalar(model, "min_value", args.min_value)
    universe_size = int(args.u) if args.u is not None else infer_histogram_size(model)
    if universe_size is None:
        raise ValueError("Could not infer U from the model. Pass --u explicitly.")

    keys = build_keys(args, min_value=min_value, universe_size=universe_size)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    histogram = session.run([output_name], {input_name: keys})[0].astype(np.int64)

    expected_histogram = np.bincount(
        keys - min_value,
        minlength=universe_size,
    ).astype(np.int64)
    sorted_from_histogram = project_histogram(histogram, min_value)

    histogram_ok = np.array_equal(histogram, expected_histogram)
    sorted_ok = np.array_equal(sorted_from_histogram, np.sort(keys))
    ordered_ok = bool(np.all(sorted_from_histogram[:-1] <= sorted_from_histogram[1:]))
    length_ok = sorted_from_histogram.size == keys.size

    limit = keys.size if args.show_full else 64
    print("=== ONNX-Accelerated-DialSort test ===")
    print(f"Model      : {model_path}")
    print(f"Input name : {input_name}")
    print(f"Output name: {output_name}")
    print(f"N          : {keys.size}")
    print(f"U          : {universe_size}")
    print(f"min_value  : {min_value}")
    print(f"Input      : {compact(keys, limit)}")
    print(f"Histogram  : {compact(histogram, limit)}")
    print(f"Sorted     : {compact(sorted_from_histogram, limit)}")
    print(f"Histogram matches numpy.bincount: {histogram_ok}")
    print(f"Projection matches numpy.sort   : {sorted_ok}")
    print(f"Projection is sorted            : {ordered_ok}")
    print(f"Length is preserved             : {length_ok}")

    if not (histogram_ok and sorted_ok and ordered_ok and length_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
