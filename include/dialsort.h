#pragma once

#include "dto/sort_result_dto.h"

#include <cstdint>
#include <vector>

class dialsort {
private:
    int64_t minValue;

    void validateInput(
        const std::vector<int64_t>& input,
        int64_t n,
        int64_t u
    ) const;

    std::vector<int64_t> projectHistogramToSortedVector(
        const std::vector<int64_t>& histogram,
        int64_t expectedN
    ) const;

public:
    explicit dialsort(int64_t minValue = 0);

    sort_result_dto sort(
        const std::vector<int64_t>& unsortedArray,
        int64_t n,
        int64_t u
    ) const;
};
