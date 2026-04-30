#include "../include/dialsort.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    constexpr int64_t MAX_U_COUNTING = 10'000'000;

    double elapsed_ms(
        const std::chrono::high_resolution_clock::time_point& start,
        const std::chrono::high_resolution_clock::time_point& end
    ) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
}

dialsort::dialsort(int64_t minValue)
    : minValue(minValue)
{
}

sort_result_dto dialsort::sort(
    const std::vector<int64_t>& unsortedArray,
    int64_t n,
    int64_t u
) const {
    validateInput(unsortedArray, n, u);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> histogram(static_cast<size_t>(u), 0);

    for (int64_t value : unsortedArray) {
        const size_t index = static_cast<size_t>(value - minValue);
        ++histogram[index];
    }

    std::vector<int64_t> sortedVector =
        projectHistogramToSortedVector(histogram, n);

    auto end = std::chrono::high_resolution_clock::now();

    sort_result_dto dto;
    dto.sortedVector = std::move(sortedVector);
    dto.algorithmUsed = "dialsort_counting_native";
    dto.n = n;
    dto.u = u;
    dto.success =
        static_cast<int64_t>(dto.sortedVector.size()) == n &&
        std::is_sorted(dto.sortedVector.begin(), dto.sortedVector.end());
    dto.elapsedMs = elapsed_ms(start, end);

    return dto;
}

void dialsort::validateInput(
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

    if (u > MAX_U_COUNTING) {
        throw std::runtime_error(
            "u exceeds the native counting DialSort domain. u=" +
            std::to_string(u) + ", max=" + std::to_string(MAX_U_COUNTING)
        );
    }

    if (static_cast<int64_t>(input.size()) != n) {
        throw std::runtime_error(
            "Input size does not match n. size=" +
            std::to_string(input.size()) + ", n=" + std::to_string(n)
        );
    }

    if (
        minValue > 0 &&
        u - 1 > std::numeric_limits<int64_t>::max() - minValue
    ) {
        throw std::runtime_error("Range [minValue, minValue + u - 1] overflows int64_t.");
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

std::vector<int64_t> dialsort::projectHistogramToSortedVector(
    const std::vector<int64_t>& histogram,
    int64_t expectedN
) const {
    std::vector<int64_t> sortedVector(static_cast<size_t>(expectedN));

    int64_t position = 0;

    for (size_t i = 0; i < histogram.size(); ++i) {
        const int64_t count = histogram[i];

        if (count < 0) {
            throw std::runtime_error("Histogram contains a negative count.");
        }

        if (position + count > expectedN) {
            throw std::runtime_error("Histogram sum exceeds n.");
        }

        const int64_t value = minValue + static_cast<int64_t>(i);

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
