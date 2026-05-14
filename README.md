# Neural DialSort

This branch contains only the files needed to keep Neural DialSort as an experimental ONNX Runtime based variant of DialSort.

It does not include the previous web application, Crow server, DTOs, frontend templates, uploaded-file handling, generated ONNX binaries, or unrelated DialSort benchmark files.

## Contents

```text
.
|-- CMakeLists.txt
|-- README.md
|-- bench_neural_dialsort.cpp
|-- docs/
|   `-- neural-dialsort.md
`-- neural_dialsort/
    |-- include/
    |   `-- neural_dialsort.h
    |-- src/
    |   `-- neural_dialsort.cpp
    |-- scripts/
    |   |-- build_dialsort_onnx.py
    |   `-- test_dialsort_onnx.py
    `-- models/
        `-- README.md
```

## What Neural DialSort Does

Neural DialSort models the DialSort histogram-ingestion step as an ONNX graph:

```text
histogram[key - min_value] += 1
```

The C++ wrapper executes the ONNX model, reads the histogram output, and projects that histogram back into a sorted `std::vector<int64_t>`.

A `NeuralDialSort` object caches ONNX Runtime sessions per universe size, so repeated benchmark calls reuse the same loaded model after warmup.

The public C++ interface is intentionally small:

```cpp
NeuralDialSort sorter;
std::vector<int64_t> values = {3, 1, 3, 5, 1, 3};

if (sorter.sort(values, 256)) {
    // values is sorted
}
```

Recoverable failures return `false`, for example when ONNX Runtime is unavailable, the model file is missing, dimensions do not match, or the universe is unsupported.

## Build Without ONNX

ONNX is disabled by default:

```bash
cmake -S . -B build
cmake --build build
```

This verifies that the branch can configure and build without requiring ONNX Runtime.

## Build The ONNX Benchmark

```bash
cmake -S . -B build-onnx -DDSORT_ENABLE_ONNX=ON -DONNXRUNTIME_DIR="/path/to/onnxruntime"
cmake --build build-onnx --config Release
```

If your ONNX Runtime package uses a nonstandard library path, pass it explicitly:

```bash
cmake -S . -B build-onnx \
  -DDSORT_ENABLE_ONNX=ON \
  -DONNXRUNTIME_DIR="/path/to/onnxruntime" \
  -DDSORT_ONNXRUNTIME_LIB="/path/to/libonnxruntime.so"
```

## Generate A Model

Install Python dependencies:

```bash
pip install numpy onnx onnxruntime
```

Generate a dynamic-length model for `U=256`:

```bash
python neural_dialsort/scripts/build_dialsort_onnx.py \
  --u 256 \
  --dynamic-n \
  --out neural_dialsort/models/dialsort_U256.onnx \
  --verify
```

Generated `.onnx` files are ignored by default.

## Test A Model

```bash
python neural_dialsort/scripts/test_dialsort_onnx.py \
  --model neural_dialsort/models/dialsort_U256.onnx \
  --array "3,1,3,5,1,3"
```

## Run The Benchmark

After building with `DSORT_ENABLE_ONNX=ON`:

```bash
./build-onnx/bench_neural_dialsort
```

On Visual Studio generators:

```bash
./build-onnx/Release/bench_neural_dialsort.exe
```

The benchmark compares:

- `NativeCounting`, a minimal native DialSort-style counting baseline
- `NeuralDialSort` through ONNX Runtime
- `std::sort`

`NativeCounting` is included only as a native CPU baseline. Performance parity for this branch is
measured against the ONNX-based Neural DialSort path from `main`, not against that baseline.

The default benchmark includes `N=1,000,000` and `N=10,000,000`, but skips `std::sort` for those large sizes to keep total runtime practical.

If a matching model is not present, the NeuralDialSort row is reported as skipped and the rest of the benchmark continues.

See [docs/neural-dialsort.md](docs/neural-dialsort.md) for the detailed design and limitations.
