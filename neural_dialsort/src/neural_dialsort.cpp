#include "neural_dialsort.h"

#include <algorithm>
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
std::filesystem::path model_path_for(const NeuralDialSortOptions& options, int64_t universe_size) {
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
    std::mutex run_mutex;
};

bool output_shape_matches(const Ort::Value& tensor, int64_t universe_size) {
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    return shape.size() == 1 && shape[0] == universe_size;
}

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
    const std::vector<int>& values,
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

    for (int value : values) {
        const int64_t widened = static_cast<int64_t>(value);
        if (widened < low || widened > high) {
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
    std::vector<int>& values
) {
    if (histogram == nullptr || histogram_size < 0) {
        return false;
    }

    std::vector<int> sorted;
    sorted.reserve(expected_size);

    for (int64_t i = 0; i < histogram_size; ++i) {
        const int64_t count = histogram[i];
        if (count < 0) {
            return false;
        }

        const auto remaining = expected_size - sorted.size();
        if (static_cast<uint64_t>(count) > static_cast<uint64_t>(remaining)) {
            return false;
        }

        const int64_t widened_value = min_value + i;
        if (
            widened_value < std::numeric_limits<int>::min() ||
            widened_value > std::numeric_limits<int>::max()
        ) {
            return false;
        }

        sorted.insert(
            sorted.end(),
            static_cast<std::size_t>(count),
            static_cast<int>(widened_value)
        );
    }

    if (sorted.size() != expected_size || !std::is_sorted(sorted.begin(), sorted.end())) {
        return false;
    }

    values = std::move(sorted);
    return true;
}
}

struct NeuralDialSort::Impl {
    explicit Impl(const NeuralDialSortOptions& created_options)
        : options(created_options),
          env(ORT_LOGGING_LEVEL_WARNING, "NeuralDialSort")
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

    NeuralDialSortOptions options;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::mutex session_mutex;
    std::unordered_map<int64_t, std::unique_ptr<CachedSession>> sessions;
};

NeuralDialSort::NeuralDialSort(const NeuralDialSortOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

NeuralDialSort::~NeuralDialSort() = default;

NeuralDialSort::NeuralDialSort(NeuralDialSort&&) noexcept = default;

NeuralDialSort& NeuralDialSort::operator=(NeuralDialSort&&) noexcept = default;

bool NeuralDialSort::sort(std::vector<int>& values, int64_t universe_size) {
    try {
        if (!impl_) {
            return false;
        }

        const NeuralDialSortOptions& options = impl_->options;

        if (values.size() <= 1) {
            return input_values_are_supported(values, options.min_value, universe_size);
        }

        if (options.intra_op_threads <= 0) {
            return false;
        }

        if (!input_values_are_supported(values, options.min_value, universe_size)) {
            return false;
        }

        CachedSession* cached = impl_->get_or_create_session(universe_size);
        if (cached == nullptr || !input_size_is_supported(*cached, values.size())) {
            return false;
        }

        std::vector<int64_t> input(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            input[i] = static_cast<int64_t>(values[i]) - options.min_value;
        }

        std::vector<int64_t> input_shape = {static_cast<int64_t>(input.size())};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input.data(),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {cached->input_name.c_str()};
        const char* output_names[] = {cached->output_name.c_str()};

        std::vector<Ort::Value> outputs;
        {
            std::lock_guard<std::mutex> lock(cached->run_mutex);
            outputs = cached->session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        }

        if (outputs.empty() || !output_shape_matches(outputs[0], universe_size)) {
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
