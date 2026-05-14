from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


def _tensor(name: str, dtype: int, dims: list[int] | list[str], vals: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=vals)


def build_dialsort_onnx(
    *,
    n: int,
    universe_size: int,
    min_value: int,
    output_path: str | Path,
    opset: int,
    dynamic_n: bool,
    mode: str,
) -> onnx.ModelProto:
    if universe_size <= 0:
        raise ValueError("--u must be greater than zero")
    if n <= 0 and not dynamic_n:
        raise ValueError("--n must be greater than zero unless --dynamic-n is set")
    if opset < 16:
        raise ValueError("Use --opset >= 16 for additive ScatterElements/ScatterND")
    if mode not in {"scatter_elements", "scatter_nd"}:
        raise ValueError("--mode must be scatter_elements or scatter_nd")

    input_shape: list[int | str] = ["N"] if dynamic_n else [n]
    keys_info = helper.make_tensor_value_info("keys", TensorProto.INT64, input_shape)
    histogram_info = helper.make_tensor_value_info(
        "histogram",
        TensorProto.INT64,
        [universe_size],
    )

    initializers = [
        _tensor("min_value", TensorProto.INT64, [], [min_value]),
        _tensor("hist_shape", TensorProto.INT64, [1], [universe_size]),
    ]

    nodes = [
        helper.make_node(
            "ConstantOfShape",
            inputs=["hist_shape"],
            outputs=["zeros"],
            value=_tensor("zero_value", TensorProto.INT64, [1], [0]),
            name="CreateZeroHistogram",
        ),
        helper.make_node(
            "Sub",
            inputs=["keys", "min_value"],
            outputs=["zero_based_keys"],
            name="MapKeyToAddress",
        ),
        helper.make_node(
            "Shape",
            inputs=["keys"],
            outputs=["keys_shape"],
            name="GetInputLength",
        ),
        helper.make_node(
            "ConstantOfShape",
            inputs=["keys_shape"],
            outputs=["ones"],
            value=_tensor("one_value", TensorProto.INT64, [1], [1]),
            name="CreateOneUpdatePerKey",
        ),
    ]

    if mode == "scatter_elements":
        nodes.append(
            helper.make_node(
                "ScatterElements",
                inputs=["zeros", "zero_based_keys", "ones"],
                outputs=["histogram"],
                axis=0,
                reduction="add",
                name="DialSortIngestion",
            )
        )
    else:
        initializers.append(_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1]))
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    inputs=["zero_based_keys", "unsqueeze_axes"],
                    outputs=["scatter_nd_indices"],
                    name="MakeScatterNDIndices",
                ),
                helper.make_node(
                    "ScatterND",
                    inputs=["zeros", "scatter_nd_indices", "ones"],
                    outputs=["histogram"],
                    reduction="add",
                    name="DialSortIngestion",
                ),
            ]
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="DialSortIngestionGraph",
        inputs=[keys_info],
        outputs=[histogram_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="neural-dialsort",
        producer_version="1.0",
        opset_imports=[helper.make_opsetid("", opset)],
        doc_string="DialSort ingestion graph: histogram[key - min_value] += 1.",
    )

    checker.check_model(model)

    try:
        model = onnx.shape_inference.infer_shapes(model)
        checker.check_model(model)
    except Exception:
        pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return model


def verify_model(model_path: str | Path, *, n: int, universe_size: int, min_value: int) -> None:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(20260321)
    keys = rng.integers(
        low=min_value,
        high=min_value + universe_size,
        size=n,
        dtype=np.int64,
    )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    histogram = session.run(["histogram"], {"keys": keys})[0]

    expected_histogram = np.bincount(
        keys - min_value,
        minlength=universe_size,
    ).astype(np.int64)

    sorted_from_histogram = np.repeat(
        np.arange(min_value, min_value + universe_size, dtype=np.int64),
        histogram,
    )

    if not np.array_equal(histogram, expected_histogram):
        raise AssertionError("ONNX histogram does not match numpy.bincount")

    if not np.array_equal(sorted_from_histogram, np.sort(keys)):
        raise AssertionError("Histogram projection does not match numpy.sort")

    print("OK: model verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ONNX graph for the DialSort histogram-ingestion step."
    )
    parser.add_argument("--n", type=int, default=8, help="Fixed input length")
    parser.add_argument("--u", type=int, required=True, help="Universe size")
    parser.add_argument("--min-value", type=int, default=0, help="Minimum key value")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .onnx path, for example neural_dialsort/models/dialsort_U256.onnx",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--dynamic-n",
        action="store_true",
        help="Use symbolic input length N instead of a fixed length",
    )
    parser.add_argument(
        "--mode",
        choices=["scatter_elements", "scatter_nd"],
        default="scatter_elements",
        help="ONNX scatter operator used for additive accumulation",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a CPU ONNX Runtime check after writing the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_dialsort_onnx(
        n=args.n,
        universe_size=args.u,
        min_value=args.min_value,
        output_path=args.out,
        opset=args.opset,
        dynamic_n=args.dynamic_n,
        mode=args.mode,
    )

    print(f"Generated ONNX model: {args.out}")
    print("Input : keys:int64[N]")
    print(f"Output: histogram:int64[{args.u}]")

    if args.verify:
        verify_model(
            args.out,
            n=args.n,
            universe_size=args.u,
            min_value=args.min_value,
        )


if __name__ == "__main__":
    main()
