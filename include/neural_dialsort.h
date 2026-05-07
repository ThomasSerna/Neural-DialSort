#pragma once

#include "dto/sort_result_dto.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

class neural_dialsort {
private:
    struct cached_model_session {
        Ort::Session session;
        std::string inputName;
        std::string outputName;
        int64_t histogramSize;

        cached_model_session(
            Ort::Session&& session,
            std::string inputName,
            std::string outputName,
            int64_t histogramSize
        ):  session(std::move(session)),
            inputName(std::move(inputName)),
            outputName(std::move(outputName)),
            histogramSize(histogramSize)
        {
        }
    };

    std::string modelDir;
    int64_t minValue;

    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::mutex cacheMutex;
    std::unordered_map<int64_t, std::unique_ptr<cached_model_session>> sessionCache;

    std::string buildModelPath(int64_t u) const;

    cached_model_session& getOrCreateSession(int64_t u);
    std::unique_ptr<cached_model_session> loadSession(int64_t u);

    void validateInput(
        const std::vector<int64_t>& input,
        int64_t n,
        int64_t u
    ) const;

    int64_t getModelHistogramSize(const Ort::Session& session) const;
    int64_t getHistogramSize(const std::vector<int64_t>& shape) const;

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
