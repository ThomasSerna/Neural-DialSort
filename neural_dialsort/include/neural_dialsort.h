#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct NeuralDialSortOptions {
    std::string model_dir = "neural_dialsort/models";
    int64_t min_value = 0;
    int intra_op_threads = 8;
};

class NeuralDialSort {
public:
    explicit NeuralDialSort(const NeuralDialSortOptions& options = {});
    ~NeuralDialSort();

    NeuralDialSort(const NeuralDialSort&) = delete;
    NeuralDialSort& operator=(const NeuralDialSort&) = delete;
    NeuralDialSort(NeuralDialSort&&) noexcept;
    NeuralDialSort& operator=(NeuralDialSort&&) noexcept;

    bool sort(std::vector<int64_t>& values, int64_t universe_size);
    bool sort_unchecked(std::vector<int64_t>& values, int64_t universe_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
