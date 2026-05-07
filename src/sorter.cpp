#include "../include/sorter.h"

#include <algorithm>
#include <random>
#include <stdexcept>

sorter::sorter()
    : normalSorter(MIN_VALUE),
      neuralSorter(MODEL_DIRECTORY, MIN_VALUE, INTRA_OP_THREADS)
{
}

sort_result_dto sorter::sort(
    const sort_request_dto& request,
    const std::string& algorithm
) {
    const std::vector<int64_t> values = generateData(request);

    if (algorithm == neuralSortName) {
        return sortNeural(values, request.u);
    }

    if (algorithm == normalSortName) {
        return sortNormal(values, request.u);
    }

    throw std::runtime_error("algorithm must be normal or neural.");
}

std::vector<sort_result_dto> sorter::compare(const sort_request_dto& request) {
    const std::vector<int64_t> values = generateData(request);

    std::vector<sort_result_dto> results;
    results.push_back(sortNormal(values, request.u));
    results.push_back(sortNeural(values, request.u));

    return results;
}

std::vector<int64_t> sorter::generateData(const sort_request_dto& request) const {
    if (request.n <= 0) {
        throw std::runtime_error("n must be positive.");
    }

    if (request.u <= 0) {
        throw std::runtime_error("u must be positive.");
    }

    if (request.distribution == "uniform") {
        return generateUniform(request.n, request.u);
    }

    if (request.distribution == "skewed") {
        return generateSkewed(request.n, request.u);
    }

    if (request.distribution == "sorted") {
        return generateSorted(request.n, request.u);
    }

    if (request.distribution == "reverse") {
        return generateReverse(request.n, request.u);
    }

    throw std::runtime_error("distribution must be uniform, skewed, sorted or reverse.");
}

std::vector<int64_t> sorter::generateUniform(int64_t n, int64_t u) const {
    std::mt19937_64 rng(SEED);
    std::uniform_int_distribution<int64_t> dist(MIN_VALUE, MIN_VALUE + u - 1);

    std::vector<int64_t> values(static_cast<size_t>(n));

    for (int64_t& value : values) {
        value = dist(rng);
    }

    return values;
}

std::vector<int64_t> sorter::generateSkewed(int64_t n, int64_t u) const {
    std::mt19937_64 rng(SEED);
    const int64_t hotLimit = std::max<int64_t>(1, u / 20);

    std::uniform_int_distribution<int64_t> hotDist(MIN_VALUE, MIN_VALUE + hotLimit - 1);
    std::uniform_int_distribution<int64_t> coldDist(MIN_VALUE, MIN_VALUE + u - 1);
    std::bernoulli_distribution pickHot(0.80);

    std::vector<int64_t> values(static_cast<size_t>(n));

    for (int64_t& value : values) {
        value = pickHot(rng) ? hotDist(rng) : coldDist(rng);
    }

    return values;
}

std::vector<int64_t> sorter::generateSorted(int64_t n, int64_t u) const {
    std::vector<int64_t> values = generateUniform(n, u);
    std::sort(values.begin(), values.end());
    return values;
}

std::vector<int64_t> sorter::generateReverse(int64_t n, int64_t u) const {
    std::vector<int64_t> values = generateSorted(n, u);
    std::reverse(values.begin(), values.end());
    return values;
}

sort_result_dto sorter::sortNormal(const std::vector<int64_t>& values, int64_t u) const {
    return normalSorter.sort(values, static_cast<int64_t>(values.size()), u);
}

sort_result_dto sorter::sortNeural(const std::vector<int64_t>& values, int64_t u) {
    return neuralSorter.sort(values, static_cast<int64_t>(values.size()), u);
}
