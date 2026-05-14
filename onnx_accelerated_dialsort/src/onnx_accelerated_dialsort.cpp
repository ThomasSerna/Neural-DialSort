#include "onnx_accelerated_dialsort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {
std::filesystem::path model_path_for(const ONNXAcceleratedDialSortOptions& options, int64_t universe_size) {
    return std::filesystem::path(options.model_dir) /
           ("dialsort_U" + std::to_string(universe_size) + ".onnx");
}

std::basic_string<ORTCHAR_T> to_ort_path(const std::filesystem::path& path) {
#ifdef _WIN32
    return path.wstring();
#else
    return path.string();
#endif
}

struct CachedSession {
    CachedSession(
        Ort::Session&& created_session,
        std::string created_input_name,
        std::string created_output_name,
        int64_t created_input_extent
    )
        : session(std::move(created_session)),
          input_name(std::move(created_input_name)),
          output_name(std::move(created_output_name)),
          input_extent(created_input_extent)
    {
    }

    Ort::Session session;
    std::string input_name;
    std::string output_name;
    int64_t input_extent = -1;
};

bool read_supported_input_extent(const Ort::Session& session, int64_t& input_extent) {
    Ort::TypeInfo input_info = session.GetInputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo shape_info = input_info.GetTensorTypeAndShapeInfo();

    if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return false;
    }

    const auto shape = shape_info.GetShape();
    if (shape.size() != 1) {
        return false;
    }

    input_extent = shape[0] < 0 ? -1 : shape[0];
    return true;
}

bool input_size_is_supported(const CachedSession& session, std::size_t input_size) {
    if (input_size > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }

    return session.input_extent < 0 ||
           session.input_extent == static_cast<int64_t>(input_size);
}

bool output_is_supported(const Ort::Session& session, int64_t universe_size) {
    Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo shape_info = output_info.GetTensorTypeAndShapeInfo();

    if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return false;
    }

    const auto shape = shape_info.GetShape();
    return shape.size() == 1 && shape[0] == universe_size;
}

bool input_values_are_supported(
    const std::vector<int64_t>& values,
    int64_t min_value,
    int64_t universe_size
) {
    if (universe_size <= 0) {
        return false;
    }

    if (universe_size - 1 > std::numeric_limits<int64_t>::max() - min_value) {
        return false;
    }

    const int64_t low = min_value;
    const int64_t high = min_value + universe_size - 1;

    for (int64_t value : values) {
        if (value < low || value > high) {
            return false;
        }
    }

    return true;
}

bool project_histogram(
    const int64_t* histogram,
    int64_t histogram_size,
    int64_t min_value,
    std::size_t expected_size,
    std::vector<int64_t>& values
) {
    if (histogram == nullptr || histogram_size < 0) {
        return false;
    }

    if (expected_size != values.size()) {
        return false;
    }

    std::size_t position = 0;

    for (int64_t i = 0; i < histogram_size; ++i) {
        const int64_t raw_count = histogram[i];
        if (raw_count < 0) {
            return false;
        }

        const std::size_t count = static_cast<std::size_t>(raw_count);
        if (count > expected_size - position) {
            return false;
        }

        const int64_t value = min_value + i;

        std::fill_n(
            values.begin() + static_cast<std::ptrdiff_t>(position),
            count,
            value
        );
        position += count;
    }

    return position == expected_size;
}
}

struct ONNXAcceleratedDialSort::Impl {
    explicit Impl(const ONNXAcceleratedDialSortOptions& created_options)
        : options(created_options),
          env(ORT_LOGGING_LEVEL_WARNING, "ONNX-Accelerated-DialSort")
    {
        if (options.intra_op_threads > 0) {
            session_options.SetIntraOpNumThreads(options.intra_op_threads);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    CachedSession* get_or_create_session(int64_t universe_size) {
        std::lock_guard<std::mutex> lock(session_mutex);

        const auto cached = sessions.find(universe_size);
        if (cached != sessions.end()) {
            return cached->second.get();
        }

        if (options.intra_op_threads <= 0) {
            return nullptr;
        }

        const std::filesystem::path model_path = model_path_for(options, universe_size);
        if (!std::filesystem::exists(model_path)) {
            return nullptr;
        }

        const auto ort_model_path = to_ort_path(model_path);
        Ort::Session session(env, ort_model_path.c_str(), session_options);

        int64_t input_extent = -1;
        if (
            !read_supported_input_extent(session, input_extent) ||
            !output_is_supported(session, universe_size)
        ) {
            return nullptr;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        auto cached_session = std::make_unique<CachedSession>(
            std::move(session),
            std::string(input_name.get()),
            std::string(output_name.get()),
            input_extent
        );
        CachedSession* result = cached_session.get();
        sessions.emplace(universe_size, std::move(cached_session));
        return result;
    }

    ONNXAcceleratedDialSortOptions options;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::mutex session_mutex;
    std::unordered_map<int64_t, std::unique_ptr<CachedSession>> sessions;
};

ONNXAcceleratedDialSort::ONNXAcceleratedDialSort(const ONNXAcceleratedDialSortOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

ONNXAcceleratedDialSort::~ONNXAcceleratedDialSort() = default;

ONNXAcceleratedDialSort::ONNXAcceleratedDialSort(ONNXAcceleratedDialSort&&) noexcept = default;

ONNXAcceleratedDialSort& ONNXAcceleratedDialSort::operator=(ONNXAcceleratedDialSort&&) noexcept = default;

bool ONNXAcceleratedDialSort::sort(std::vector<int64_t>& values, int64_t universe_size) {
    if (!impl_) {
        return false;
    }

    const ONNXAcceleratedDialSortOptions& options = impl_->options;

    if (!input_values_are_supported(values, options.min_value, universe_size)) {
        return false;
    }

    return sort_unchecked(values, universe_size);
}

bool ONNXAcceleratedDialSort::sort_unchecked(std::vector<int64_t>& values, int64_t universe_size) {
    try {
        if (!impl_) {
            return false;
        }

        const ONNXAcceleratedDialSortOptions& options = impl_->options;

        if (options.intra_op_threads <= 0 || universe_size <= 0) {
            return false;
        }

        if (values.size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }

        CachedSession* cached = impl_->get_or_create_session(universe_size);
        if (cached == nullptr || !input_size_is_supported(*cached, values.size())) {
            return false;
        }

        std::array<int64_t, 1> input_shape = {static_cast<int64_t>(values.size())};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            values.data(),
            values.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {cached->input_name.c_str()};
        const char* output_names[] = {cached->output_name.c_str()};

        std::vector<Ort::Value> outputs;
        outputs = cached->session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        if (outputs.empty()) {
            return false;
        }

        const int64_t* histogram = outputs[0].GetTensorData<int64_t>();
        return project_histogram(
            histogram,
            universe_size,
            options.min_value,
            values.size(),
            values
        );
    } catch (...) {
        return false;
    }
}
