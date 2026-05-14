# ONNX models

This directory is intended for generated ONNX-Accelerated-DialSort models.

Generated `.onnx` files are not committed by default.

Generate a model with:

```bash
python onnx_accelerated_dialsort/scripts/build_dialsort_onnx.py --u 256 --dynamic-n --out onnx_accelerated_dialsort/models/dialsort_U256.onnx
```

Then enable the ONNX benchmark through CMake:

```bash
cmake -S . -B build-onnx -DDSORT_ENABLE_ONNX=ON -DONNXRUNTIME_DIR="/path/to/onnxruntime"
cmake --build build-onnx --config Release
```
