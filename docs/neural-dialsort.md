# Neural DialSort

Neural DialSort is an experimental ONNX Runtime variant of DialSort. It is intended for benchmarking and architecture experiments, not as a replacement for the native C++ DialSort implementation.

The native implementation remains the default path. ONNX support is disabled by default and is only built when `DSORT_ENABLE_ONNX=ON` is passed to CMake.

## What the ONNX graph does

DialSort has two major phases:

1. Ingestion: build a histogram with `H[key - min_value] += 1`.
2. Projection: scan the histogram and write the sorted values back.

Neural DialSort models the ingestion phase as an ONNX graph. The graph receives a 1D `int64` tensor named `keys` and returns a 1D `int64` tensor named `histogram` with length `U`.

The C++ wrapper then projects that histogram back into the caller's `std::vector<int64_t>`.

Each `NeuralDialSort` instance keeps an ONNX Runtime session cache keyed by universe size. The
first call for a given `U` loads and validates `dialsort_U<U>.onnx`; later calls reuse that
session so benchmark warmup rounds measure inference and projection rather than repeated model
loading.

## C++ interface

The reusable C++ interface is intentionally small and independent of any web application DTOs:

```cpp
#include "neural_dialsort.h"

NeuralDialSort sorter;
std::vector<int64_t> values = {3, 1, 3, 5, 1, 3};

if (sorter.sort(values, 256)) {
    // values is sorted
}
```

`NeuralDialSort::sort` returns `true` only when the model is present, ONNX Runtime executes successfully, the model dimensions match the requested universe size, and the projected output is sorted. Recoverable failures return `false` instead of throwing to the caller.

## Generate a model

Install the Python dependencies needed by the scripts:

```bash
pip install numpy onnx onnxruntime
```

Generate a model for universe `U=256`:

```bash
python neural_dialsort/scripts/build_dialsort_onnx.py \
  --u 256 \
  --dynamic-n \
  --out neural_dialsort/models/dialsort_U256.onnx \
  --verify
```

Useful arguments:

| Argument | Description |
|---|---|
| `--u` | Universe size. The model file is expected to match the benchmark's `U`. |
| `--n` | Fixed input size when `--dynamic-n` is not used. |
| `--dynamic-n` | Uses a symbolic input length, allowing different `N` values. |
| `--min-value` | Minimum valid key. The C++ default is `0`. |
| `--mode` | Uses either `scatter_elements` or `scatter_nd` for accumulation. |
| `--verify` | Runs a CPU ONNX Runtime check after model generation. |

Generated `.onnx` files are ignored by default.

## Test a model

```bash
python neural_dialsort/scripts/test_dialsort_onnx.py \
  --model neural_dialsort/models/dialsort_U256.onnx \
  --array "3,1,3,5,1,3"
```

Or test random input:

```bash
python neural_dialsort/scripts/test_dialsort_onnx.py \
  --model neural_dialsort/models/dialsort_U256.onnx \
  --random-n 10000
```

Large random model checks are supported with the same dynamic-N model:

```bash
python neural_dialsort/scripts/test_dialsort_onnx.py \
  --model neural_dialsort/models/dialsort_U1024.onnx \
  --random-n 1000000

python neural_dialsort/scripts/test_dialsort_onnx.py \
  --model neural_dialsort/models/dialsort_U1024.onnx \
  --random-n 10000000
```

The test script checks the ONNX histogram against `numpy.bincount`, projects the histogram back to a sorted vector, and compares that projection with `numpy.sort`.

## Build without ONNX

The normal build does not require ONNX Runtime:

```bash
cmake -S . -B build
cmake --build build
```

## Build with ONNX

Pass the ONNX Runtime root directory explicitly:

```bash
cmake -S . -B build-onnx \
  -DDSORT_ENABLE_ONNX=ON \
  -DONNXRUNTIME_DIR="/path/to/onnxruntime"

cmake --build build-onnx --config Release
```

By default CMake looks for:

| Platform | Default library path |
|---|---|
| Windows | `${ONNXRUNTIME_DIR}/lib/onnxruntime.lib` |
| Linux | `${ONNXRUNTIME_DIR}/lib/libonnxruntime.so` |
| macOS | `${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib` |

If your ONNX Runtime package has a different layout, pass `DSORT_ONNXRUNTIME_LIB`:

```bash
cmake -S . -B build-onnx \
  -DDSORT_ENABLE_ONNX=ON \
  -DONNXRUNTIME_DIR="/path/to/onnxruntime" \
  -DDSORT_ONNXRUNTIME_LIB="/path/to/libonnxruntime.so"
```

## Run the benchmark

After an ONNX-enabled build:

```bash
./build-onnx/bench_neural_dialsort
```

On multi-config generators such as Visual Studio:

```bash
./build-onnx/Release/bench_neural_dialsort.exe
```

The benchmark compares:

- `NativeCounting`, a native DialSort-style counting and projection baseline
- NeuralDialSort through ONNX Runtime
- `std::sort`

`NativeCounting` is a native CPU baseline, not the Neural DialSort parity target. Use the
`NeuralDialSort` rows when comparing this branch with the ONNX-based implementation from `main`.

By default it runs `N=10,000`, `N=100,000`, `N=1,000,000`, and `N=10,000,000` across the configured universe sizes and distributions. The large cases are intentionally included in the default benchmark to expose whether ONNX Runtime overhead amortizes at larger input sizes. `std::sort` is skipped for `N >= 1,000,000` because it dominates the benchmark runtime.

Each row validates correctness with `std::is_sorted`. If a matching model is missing or the model cannot run, the NeuralDialSort row is reported as skipped and the rest of the benchmark continues.

## Limitations

- This is an experimental variant for measurement and exploration.
- It currently expects one generated model per universe size, named `dialsort_U<U>.onnx`.
- The current C++ wrapper expects `int64` model input tensors and `int64` histogram output tensors, matching the implementation from the web branch.
- The public API accepts `std::vector<int64_t>` so the wrapper can pass caller storage directly to ONNX Runtime without widening or narrowing conversions.
- The ONNX graph only models histogram ingestion. Projection remains ordinary C++.
- The first call for a universe size pays the ONNX model load/session creation cost; reuse the same `NeuralDialSort` instance to benefit from the session cache.
- ONNX Runtime is a build-time and run-time dependency only when `DSORT_ENABLE_ONNX=ON`.
- Missing models, unsupported dimensions, invalid value ranges, and inference failures return `false` from `NeuralDialSort::sort`.
