#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "../include/neural_dialsort.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <windows.h>

namespace {
    double elapsed_ms(
        const std::chrono::high_resolution_clock::time_point& start,
        const std::chrono::high_resolution_clock::time_point& end
    ) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::wstring utf8_to_wide(const std::string& text) {
        if (text.empty()) {
            return std::wstring();
        }

        const int inputSize = static_cast<int>(text.size());
        const int wideSize = MultiByteToWideChar(
            CP_UTF8,
            0,
            text.c_str(),
            inputSize,
            nullptr,
            0
        );

        if (wideSize <= 0) {
            throw std::runtime_error("Could not convert model path to UTF-16.");
        }

        std::wstring result(static_cast<size_t>(wideSize), L'\0');

        const int converted = MultiByteToWideChar(
            CP_UTF8,
            0,
            text.c_str(),
            inputSize,
            result.data(),
            wideSize
        );

        if (converted <= 0) {
            throw std::runtime_error("Model path conversion to UTF-16 failed.");
        }

        return result;
    }
}

neural_dialsort::neural_dialsort(
    const std::string& modelDirectory,
    int64_t minValue,
    int intraOpThreads
)
    : modelDir(modelDirectory),
      minValue(minValue),
      env(ORT_LOGGING_LEVEL_WARNING, "neural_dialsort")
{
    if (intraOpThreads <= 0) {
        throw std::runtime_error("intraOpThreads must be positive.");
    }

    sessionOptions.SetIntraOpNumThreads(intraOpThreads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

sort_result_dto neural_dialsort::sort(
    const std::vector<int64_t>& unsortedArray,
    int64_t n,
    int64_t u
) {
    validateInput(unsortedArray, n, u);

    const std::string modelPath = buildModelPath(u);

    if (!std::filesystem::exists(modelPath)) {
        throw std::runtime_error(
            "Expected model does not exist: " + modelPath +
            ". The model must be named dialsort_U" + std::to_string(u) + ".onnx"
        );
    }

    auto start = std::chrono::high_resolution_clock::now();

    const std::wstring wideModelPath = utf8_to_wide(modelPath);
    Ort::Session session(env, wideModelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);

    const char* inputName = inputNameAllocated.get();
    const char* outputName = outputNameAllocated.get();

    std::vector<const char*> inputNames = {inputName};
    std::vector<const char*> outputNames = {outputName};
    std::vector<int64_t> inputShape = {n};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo,
        const_cast<int64_t*>(unsortedArray.data()),
        unsortedArray.size(),
        inputShape.data(),
        inputShape.size()
    );

    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        1,
        outputNames.data(),
        1
    );

    if (outputs.empty()) {
        throw std::runtime_error("The ONNX model did not produce any output.");
    }

    const int64_t histogramSize = getHistogramSize(outputs[0]);

    if (histogramSize != u) {
        throw std::runtime_error(
            "Model histogram has U=" + std::to_string(histogramSize) +
            ", but sort received u=" + std::to_string(u)
        );
    }

    const int64_t* histogram = outputs[0].GetTensorData<int64_t>();

    std::vector<int64_t> sortedVector =
        projectHistogramToSortedVector(histogram, histogramSize, n);

    auto end = std::chrono::high_resolution_clock::now();

    sort_result_dto dto;
    dto.sortedVector = std::move(sortedVector);
    dto.algorithmUsed = "neural_dialsort";
    dto.n = n;
    dto.u = u;
    dto.success =
        static_cast<int64_t>(dto.sortedVector.size()) == n &&
        std::is_sorted(dto.sortedVector.begin(), dto.sortedVector.end());
    dto.elapsedMs = elapsed_ms(start, end);

    return dto;
}

std::string neural_dialsort::buildModelPath(int64_t u) const {
    return modelDir + "/dialsort_U" + std::to_string(u) + ".onnx";
}

void neural_dialsort::validateInput(
    const std::vector<int64_t>& input,
    int64_t n,
    int64_t u
) const {
    if (n <= 0) {
        throw std::runtime_error("n must be positive.");
    }

    if (u <= 0) {
        throw std::runtime_error("u must be positive.");
    }

    if (static_cast<int64_t>(input.size()) != n) {
        throw std::runtime_error(
            "Input size does not match n. size=" +
            std::to_string(input.size()) + ", n=" + std::to_string(n)
        );
    }

    const int64_t low = minValue;
    const int64_t high = minValue + u - 1;

    for (int64_t value : input) {
        if (value < low || value > high) {
            throw std::runtime_error(
                "Value outside allowed range [" +
                std::to_string(low) + ", " + std::to_string(high) +
                "]: " + std::to_string(value)
            );
        }
    }
}

int64_t neural_dialsort::getHistogramSize(const Ort::Value& outputTensor) const {
    Ort::TensorTypeAndShapeInfo info = outputTensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();

    if (shape.size() != 1 || shape[0] <= 0) {
        throw std::runtime_error("Model output must be a 1D tensor: histogram[U].");
    }

    return shape[0];
}

std::vector<int64_t> neural_dialsort::projectHistogramToSortedVector(
    const int64_t* histogram,
    int64_t histogramSize,
    int64_t expectedN
) const {
    if (histogram == nullptr) {
        throw std::runtime_error("histogram is nullptr.");
    }

    std::vector<int64_t> sortedVector(static_cast<size_t>(expectedN));

    int64_t position = 0;

    for (int64_t i = 0; i < histogramSize; ++i) {
        const int64_t count = histogram[i];

        if (count < 0) {
            throw std::runtime_error("Histogram contains a negative count.");
        }

        if (position + count > expectedN) {
            throw std::runtime_error("Histogram sum exceeds n.");
        }

        const int64_t value = minValue + i;

        std::fill_n(
            sortedVector.begin() + static_cast<std::ptrdiff_t>(position),
            static_cast<size_t>(count),
            value
        );

        position += count;
    }

    if (position != expectedN) {
        throw std::runtime_error("Histogram sum does not match n.");
    }

    return sortedVector;
}
