from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
from onnx import TensorProto, checker, helper


ScatterMode = Literal["scatter_elements", "scatter_nd"]


def _tensor(name: str, dtype: int, dims: list[int] | list[str] | tuple, vals: list[int]):
    return helper.make_tensor(name=name, data_type=dtype, dims=list(dims), vals=vals)


def build_dialsort_onnx(
    *,
    n: int,
    universe_size: int,
    min_value: int,
    output_path: str | Path,
    opset: int = 18,
    dynamic_n: bool = False,
    mode: ScatterMode = "scatter_elements",
) -> onnx.ModelProto:

    if universe_size <= 0:
        raise ValueError("universe_size debe ser mayor que 0")
    if n <= 0 and not dynamic_n:
        raise ValueError("n debe ser mayor que 0 cuando dynamic_n=False")
    if opset < 16:
        raise ValueError("Usa opset >= 16 para reduction='add' en ScatterElements/ScatterND")
    if mode not in ("scatter_elements", "scatter_nd"):
        raise ValueError("mode debe ser 'scatter_elements' o 'scatter_nd'")

    input_shape: list[int | str] = ["N"] if dynamic_n else [n]

    keys_info = helper.make_tensor_value_info("keys", TensorProto.INT64, input_shape)
    hist_info = helper.make_tensor_value_info(
        "histogram", TensorProto.INT64, [universe_size]
    )

    # Initializers: constantes del grafo.
    initializers = [
        _tensor("min_value", TensorProto.INT64, [], [int(min_value)]),
        _tensor("hist_shape", TensorProto.INT64, [1], [int(universe_size)]),
    ]

    zero_value = _tensor("zero_value", TensorProto.INT64, [1], [0])
    one_value = _tensor("one_value", TensorProto.INT64, [1], [1])

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "ConstantOfShape",
            inputs=["hist_shape"],
            outputs=["zeros"],
            value=zero_value,
            name="CreateZeroHistogram",
        ),
        helper.make_node(
            "Sub",
            inputs=["keys", "min_value"],
            outputs=["zero_based_keys"],
            name="MapKeyToAddress_k_minus_min",
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
            value=one_value,
            name="CreateOneUpdatePerKey",
        ),
    ]

    if mode == "scatter_elements":
        # 1D exacto:
        # histogram[index[i]] += ones[i]
        nodes.append(
            helper.make_node(
                "ScatterElements",
                inputs=["zeros", "zero_based_keys", "ones"],
                outputs=["histogram"],
                axis=0,
                reduction="add",
                name="DialSortIngestion_H_index_add_1",
            )
        )
    else:
        # ScatterND requiere indices con forma [N, 1] para un tensor data de forma [U].
        initializers.append(_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1]))
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    inputs=["zero_based_keys", "unsqueeze_axes"],
                    outputs=["scatter_nd_indices"],
                    name="MakeScatterNDIndices_Nx1",
                ),
                helper.make_node(
                    "ScatterND",
                    inputs=["zeros", "scatter_nd_indices", "ones"],
                    outputs=["histogram"],
                    reduction="add",
                    name="DialSortIngestion_H_index_add_1",
                ),
            ]
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="DialSort_AutoIndex_Ingestion",
        inputs=[keys_info],
        outputs=[hist_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="dialsort-autoindex-generator",
        producer_version="1.0",
        opset_imports=[helper.make_opsetid("", opset)],
        doc_string=(
            "Deterministic ONNX graph for DialSort ingestion: "
            "H[k - min_value] += 1. Outputs the canonical histogram H."
        ),
    )

    checker.check_model(model)

    # La inferencia de formas es util para inspeccion con Netron/ORT.
    try:
        model = onnx.shape_inference.infer_shapes(model)
        checker.check_model(model)
    except Exception:
        pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return model


def project_histogram(histogram: np.ndarray, min_value: int = 0) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.int64)
    values = np.arange(min_value, min_value + histogram.shape[0], dtype=np.int64)
    return np.repeat(values, histogram)


def verify_model(model_path: str | Path, *, n: int, universe_size: int, min_value: int) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime no esta instalado."
        ) from exc

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

    if not np.array_equal(histogram, expected_histogram):
        raise AssertionError("El histograma ONNX no coincide con np.bincount")

    projected = project_histogram(histogram, min_value=min_value)
    expected_sorted = np.sort(keys)

    if not np.array_equal(projected, expected_sorted):
        raise AssertionError("La proyeccion H -> arreglo ordenado no coincide con np.sort")

    print("OK: modelo verificado")
    print(f"  model      : {model_path}")
    print(f"  N          : {n}")
    print(f"  U          : {universe_size}")
    print(f"  min_value  : {min_value}")
    print(f"  input      : {keys.tolist() if n <= 32 else str(keys[:32].tolist()) + ' ...'}")
    print(f"  histogram  : {histogram.tolist() if universe_size <= 64 else str(histogram[:64].tolist()) + ' ...'}")
    print(f"  sorted     : {projected.tolist() if n <= 32 else str(projected[:32].tolist()) + ' ...'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un modelo ONNX determinista para DialSort por auto-indexacion."
    )
    parser.add_argument("--n", type=int, default=8, help="Longitud fija de entrada N")
    parser.add_argument("--u", type=int, default=6, help="Tamano del universo U")
    parser.add_argument(
        "--min-value",
        type=int,
        default=0,
        help="Valor minimo del universo; el indice usado es k - min_value",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="dialsort_autoindex.onnx",
        help="Ruta de salida del modelo ONNX",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Version del opset ONNX. Recomendado: 18",
    )
    parser.add_argument(
        "--dynamic-n",
        action="store_true",
        help="Usa dimension simbolica N en vez de una longitud fija",
    )
    parser.add_argument(
        "--mode",
        choices=["scatter_elements", "scatter_nd"],
        default="scatter_elements",
        help="Operador ONNX para acumulacion aditiva",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Ejecuta una prueba con onnxruntime despues de generar el modelo",
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

    print(f"Modelo ONNX generado: {args.out}")
    print("Entrada : keys:int64[N]")
    print(f"Salida  : histogram:int64[{args.u}]")
    print("Operacion codificada: histogram[keys[i] - min_value] += 1")

    if args.verify:
        verify_model(args.out, n=args.n, universe_size=args.u, min_value=args.min_value)


if __name__ == "__main__":
    main()
