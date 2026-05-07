#pragma once

#include "dto/sort_request_dto.h"
#include "dto/sort_result_dto.h"
#include "dialsort.h"
#include "neural_dialsort.h"

#include <cstdint>
#include <string>
#include <vector>

class sorter {
public:
    const std::string normalSortName = "normal";
    const std::string neuralSortName = "neural";

    sorter();

    sort_result_dto sort(const sort_request_dto& request, const std::string& algorithm);
    std::vector<sort_result_dto> compare(const sort_request_dto& request);

private:
    static constexpr int64_t MIN_VALUE = 0;
    static constexpr uint64_t SEED = 123;
    static constexpr const char* MODEL_DIRECTORY = "model";
    static constexpr int INTRA_OP_THREADS = 8;

    dialsort normalSorter;
    neural_dialsort neuralSorter;

    std::vector<int64_t> generateData(const sort_request_dto& request) const;
    std::vector<int64_t> generateUniform(int64_t n, int64_t u) const;
    std::vector<int64_t> generateSkewed(int64_t n, int64_t u) const;
    std::vector<int64_t> generateSorted(int64_t n, int64_t u) const;
    std::vector<int64_t> generateReverse(int64_t n, int64_t u) const;

    sort_result_dto sortNormal(const std::vector<int64_t>& values, int64_t u) const;
    sort_result_dto sortNeural(const std::vector<int64_t>& values, int64_t u);
};
