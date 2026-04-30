#pragma once

#include "dto/sort_result_dto.h"

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class neural_dialsort {
private:
    std::string modelDir;
    int64_t minValue;

    Ort::Env env;
    Ort::SessionOptions sessionOptions;

    std::string buildModelPath(int64_t u) const;

    void validateInput(
        const std::vector<int64_t>& input,
        int64_t n,
        int64_t u
    ) const;

    int64_t getHistogramSize(const Ort::Value& outputTensor) const;

    std::vector<int64_t> projectHistogramToSortedVector(
        const int64_t* histogram,
        int64_t histogramSize,
        int64_t expectedN
    ) const;

public:
    explicit neural_dialsort(
        const std::string& modelDirectory = "model",
        int64_t minValue = 0,
        int intraOpThreads = 8
    );

    sort_result_dto sort(
        const std::vector<int64_t>& unsortedArray,
        int64_t n,
        int64_t u
    );
};
