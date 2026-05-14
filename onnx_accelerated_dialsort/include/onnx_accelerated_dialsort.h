#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ONNXAcceleratedDialSortOptions {
    std::string model_dir = "onnx_accelerated_dialsort/models";
    int64_t min_value = 0;
    int intra_op_threads = 8;
};

class ONNXAcceleratedDialSort {
public:
    explicit ONNXAcceleratedDialSort(const ONNXAcceleratedDialSortOptions& options = {});
    ~ONNXAcceleratedDialSort();

    ONNXAcceleratedDialSort(const ONNXAcceleratedDialSort&) = delete;
    ONNXAcceleratedDialSort& operator=(const ONNXAcceleratedDialSort&) = delete;
    ONNXAcceleratedDialSort(ONNXAcceleratedDialSort&&) noexcept;
    ONNXAcceleratedDialSort& operator=(ONNXAcceleratedDialSort&&) noexcept;

    bool sort(std::vector<int64_t>& values, int64_t universe_size);
    bool sort_unchecked(std::vector<int64_t>& values, int64_t universe_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
