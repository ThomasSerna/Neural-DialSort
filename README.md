# Neural DialSort

Neural DialSort is an experimental C++ implementation of DialSort that compares a native counting-based version against a neural/ONNX-based variant.

The project exposes a small HTTP API built with Crow. Through this API, it can generate input arrays with different distributions, sort them using either the native DialSort implementation or the ONNX-based implementation, and return execution metadata such as elapsed time, input size, universe size, algorithm used, and success status.

## Main idea

DialSort sorts integer keys by treating each value as an address inside a histogram.

Instead of comparing elements directly, the algorithm counts how many times each value appears:

```text
H[value - min_value] += 1
```

Then, the sorted vector is reconstructed by projecting the histogram back into an ordered array.

This repository contains two variants:

| Algorithm | Description |
|---|---|
| `normal` | Native C++ counting-based DialSort implementation |
| `neural` | ONNX Runtime implementation that uses a pre-generated DialSort model |

## Features

- Native C++ DialSort implementation.
- ONNX Runtime based DialSort implementation.
- HTTP server using Crow.
- API endpoint to run a single algorithm.
- API endpoint to compare both algorithms.
- Synthetic data generation with multiple distributions.
- Python scripts to generate and test ONNX models.
- Experimental NPU testing script using ONNX Runtime providers.

## Project structure

```text
Neural-DialSort/
│
├── external/
│   └── Third-party headers, including Crow
│
├── include/
│   ├── dto/
│   │   ├── sort_request_dto.h
│   │   └── sort_result_dto.h
│   ├── dialsort.h
│   ├── neural_dialsort.h
│   ├── server.h
│   └── sorter.h
│
├── model/
│   ├── build_dialsort_onnx.py
│   ├── test_dialsort_onnx.py
│   ├── test_dialsort_onnx_npu.py
│   ├── dialsort_U256.onnx
│   ├── dialsort_U1000.onnx
│   ├── dialsort_U65536.onnx
│   └── dialsort_U65536_npu.onnx
│
├── src/
│   ├── dialsort.cpp
│   ├── main.cpp
│   ├── neural_dialsort.cpp
│   ├── server.cpp
│   └── sorter.cpp
│
├── CMakeLists.txt
└── README.md
```

## Requirements

### C++

- CMake 3.20 or newer
- C++20 compatible compiler
- ONNX Runtime for C++
- Windows environment recommended, since the current CMake configuration links against:
  - `onnxruntime.lib`
  - `ws2_32`
  - `mswsock`

### Python

Only needed if you want to generate or test ONNX models.

Recommended packages:

```bash
pip install numpy onnx onnxruntime
```

For NPU experiments, you need an ONNX Runtime installation that exposes the required NPU execution provider, such as `VitisAIExecutionProvider`.

## Building the C++ project

The project requires the `ONNXRUNTIME_DIR` variable to point to the root directory of your ONNX Runtime installation.

Example using CMake:

```bash
cmake -S . -B build -DONNXRUNTIME_DIR="C:/path/to/onnxruntime"
cmake --build build --config Release
```

Example on Windows after building:

```bash
./build/Release/neuralDialsort.exe
```

The server starts on port `8080`.

```text
http://localhost:8080
```

## ONNX model behavior

The ONNX model represents the histogram-building phase of DialSort.

Given an input vector:

```text
keys = [3, 1, 3, 5, 1, 3]
```

The model computes a histogram:

```text
H[key - min_value] += 1
```

Then the C++ implementation projects that histogram back into a sorted vector.

## Generating an ONNX model

You can generate a new DialSort ONNX model with:

```bash
python model/build_dialsort_onnx.py --n 8 --u 256 --out model/dialsort_U256.onnx --verify
```

Useful arguments:

| Argument | Description |
|---|---|
| `--n` | Fixed input size |
| `--u` | Universe size |
| `--min-value` | Minimum valid value |
| `--out` | Output model path |
| `--dynamic-n` | Uses a symbolic input size instead of a fixed one |
| `--mode` | ONNX scatter mode: `scatter_elements` or `scatter_nd` |
| `--verify` | Runs an ONNX Runtime verification after generating the model |

Example with dynamic input length:

```bash
python model/build_dialsort_onnx.py ^
  --n 8 ^
  --u 1000 ^
  --dynamic-n ^
  --out model/dialsort_U1000.onnx ^
  --verify
```

## Testing an ONNX model on CPU

```bash
python model/test_dialsort_onnx.py --model model/dialsort_U256.onnx --array "3,1,3,5,1,3"
```

You can also generate random input:

```bash
python model/test_dialsort_onnx.py --model model/dialsort_U256.onnx --random-n 32
```

The script validates:

- ONNX histogram equals `np.bincount`.
- Histogram projection equals `np.sort`.
- The projected vector is sorted.
- The output length matches the original input length.

## Testing with NPU provider

The repository includes an experimental script for testing ONNX Runtime providers:

```bash
python model/test_dialsort_onnx_npu.py ^
  --model model/dialsort_U65536_npu.onnx ^
  --random-n 1024 ^
  --provider npu_cpu
```

Available provider modes:

| Provider | Description |
|---|---|
| `cpu` | Uses `CPUExecutionProvider` |
| `npu` | Uses `VitisAIExecutionProvider` |
| `npu_cpu` | Tries `VitisAIExecutionProvider` with CPU fallback |

To force Phoenix/Hawk Point specific provider options, define:

```bash
set RYZEN_AI_NPU_TYPE=PHX_HPT
```

For Ryzen AI 300 / Strix / Krackan systems, the script uses simpler provider options by default.

## Technologies used

- C++20
- CMake
- Crow
- ONNX
- ONNX Runtime
- Python
- NumPy

## Author

- Thomas Serna Saldarriaga
- David Alzate Monroy