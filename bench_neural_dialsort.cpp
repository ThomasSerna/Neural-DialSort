#include "neural_dialsort.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int WARMUP_ROUNDS = 10;
constexpr int MEASURE_ROUNDS = 3;
constexpr uint64_t SEED = 20260321ULL;
constexpr int64_t MAX_U_COUNTING = 10'000'000;

int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

bool dialsort_counting(std::vector<int>& values, int universe_size) {
    if (values.size() <= 1) {
        return true;
    }

    if (universe_size <= 0 || universe_size > MAX_U_COUNTING) {
        return false;
    }

    std::vector<int64_t> histogram(static_cast<std::size_t>(universe_size), 0);
    for (int value : values) {
        if (value < 0 || value >= universe_size) {
            return false;
        }
        ++histogram[static_cast<std::size_t>(value)];
    }

    std::size_t out = 0;
    for (int value = 0; value < universe_size; ++value) {
        const int64_t count = histogram[static_cast<std::size_t>(value)];
        for (int64_t i = 0; i < count; ++i) {
            values[out++] = value;
        }
    }

    return out == values.size();
}

std::vector<int> gen_uniform(std::size_t n, int universe_size, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, universe_size - 1);
    std::vector<int> values(n);
    for (int& value : values) {
        value = dist(rng);
    }
    return values;
}

std::vector<int> gen_skewed(std::size_t n, int universe_size, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int hot_limit = std::max(1, universe_size / 20);
    std::uniform_int_distribution<int> hot(0, hot_limit - 1);
    std::uniform_int_distribution<int> cold(0, universe_size - 1);
    std::bernoulli_distribution pick_hot(0.80);

    std::vector<int> values(n);
    for (int& value : values) {
        value = pick_hot(rng) ? hot(rng) : cold(rng);
    }
    return values;
}

std::vector<int> gen_sorted(std::size_t n, int universe_size, uint64_t seed) {
    auto values = gen_uniform(n, universe_size, seed);
    std::sort(values.begin(), values.end());
    return values;
}

std::vector<int> gen_reverse(std::size_t n, int universe_size, uint64_t seed) {
    auto values = gen_sorted(n, universe_size, seed);
    std::reverse(values.begin(), values.end());
    return values;
}

using SortFn = std::function<bool(std::vector<int>&)>;

struct Row {
    std::string algorithm;
    std::string distribution;
    std::size_t n = 0;
    int universe_size = 0;
    int64_t best_ns = 0;
    double ms = 0.0;
    double mkeys_s = 0.0;
    double speedup_vs_std_sort = 0.0;
    bool correct = false;
    bool skipped = false;
};

Row run_one(
    const std::string& algorithm,
    const std::string& distribution,
    const std::vector<int>& base,
    int universe_size,
    const SortFn& sort_fn,
    int64_t std_sort_ns
) {
    Row row;
    row.algorithm = algorithm;
    row.distribution = distribution;
    row.n = base.size();
    row.universe_size = universe_size;

    for (int i = 0; i < WARMUP_ROUNDS; ++i) {
        auto tmp = base;
        if (!sort_fn(tmp) || !std::is_sorted(tmp.begin(), tmp.end())) {
            row.skipped = true;
            return row;
        }
    }

    int64_t best = std::numeric_limits<int64_t>::max();
    bool correct = true;

    for (int i = 0; i < MEASURE_ROUNDS; ++i) {
        auto tmp = base;
        const int64_t start = now_ns();
        const bool ran = sort_fn(tmp);
        const int64_t elapsed = now_ns() - start;

        if (!ran) {
            row.skipped = true;
            return row;
        }

        correct = correct && std::is_sorted(tmp.begin(), tmp.end());
        if (elapsed > 0 && elapsed < best) {
            best = elapsed;
        }
    }

    if (best == std::numeric_limits<int64_t>::max()) {
        row.skipped = true;
        return row;
    }

    row.best_ns = best;
    row.ms = static_cast<double>(best) / 1e6;
    row.mkeys_s = (static_cast<double>(base.size()) / (static_cast<double>(best) / 1e9)) / 1e6;
    row.speedup_vs_std_sort =
        std_sort_ns > 0 ? static_cast<double>(std_sort_ns) / static_cast<double>(best) : 0.0;
    row.correct = correct;
    return row;
}

void print_header() {
    std::cout << std::left
              << std::setw(22) << "Algorithm"
              << std::setw(14) << "Distribution"
              << std::setw(12) << "N"
              << std::setw(8) << "U"
              << std::setw(12) << "ms"
              << std::setw(14) << "M keys/s"
              << std::setw(14) << "vs std::sort"
              << "OK\n";
    std::cout << std::string(96, '-') << "\n";
}

void print_row(const Row& row) {
    if (row.skipped) {
        std::cout << std::left
                  << std::setw(22) << row.algorithm
                  << std::setw(14) << row.distribution
                  << std::setw(12) << row.n
                  << std::setw(8) << row.universe_size
                  << "SKIPPED";
        if (row.algorithm == "NeuralDialSort") {
            std::cout << " (model/runtime unavailable)";
        }
        std::cout << "\n";
        return;
    }

    std::cout << std::left
              << std::setw(22) << row.algorithm
              << std::setw(14) << row.distribution
              << std::setw(12) << row.n
              << std::setw(8) << row.universe_size
              << std::fixed << std::setprecision(3)
              << std::setw(12) << row.ms
              << std::setw(14) << row.mkeys_s
              << std::setw(14) << row.speedup_vs_std_sort
              << (row.correct ? "OK" : "FAIL") << "\n";
}

struct Dist {
    std::string name;
    std::function<std::vector<int>(std::size_t, int, uint64_t)> generate;
};

struct Args {
    std::string model_dir = "neural_dialsort/models";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string current = argv[i];
        if (current == "--model-dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (current == "--help" || current == "-h") {
            std::cout << "Usage: bench_neural_dialsort [--model-dir PATH]\n";
            std::exit(0);
        }
    }
    return args;
}
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    NeuralDialSort neural({
        args.model_dir,
        0,
        8,
    });

    const std::vector<std::size_t> ns = {10'000, 100'000};
    const std::vector<int> universe_sizes = {256, 1024, 65536};
    const std::vector<Dist> distributions = {
        {"uniform", gen_uniform},
        {"skewed", gen_skewed},
        {"sorted", gen_sorted},
        {"reverse", gen_reverse},
    };

    std::cout << "Neural DialSort optional ONNX benchmark\n";
    std::cout << "Model directory: " << args.model_dir << "\n";
    std::cout << "Warmup rounds: " << WARMUP_ROUNDS << "\n";
    std::cout << "Measurement rounds: best of " << MEASURE_ROUNDS << "\n\n";

    print_header();

    for (std::size_t n : ns) {
        for (int universe_size : universe_sizes) {
            for (const auto& dist : distributions) {
                const uint64_t seed = SEED ^
                                      (static_cast<uint64_t>(n) * 1000003ULL) ^
                                      (static_cast<uint64_t>(universe_size) * 7919ULL);
                const auto base = dist.generate(n, universe_size, seed);

                const SortFn std_sort = [](std::vector<int>& values) {
                    std::sort(values.begin(), values.end());
                    return true;
                };

                Row std_row = run_one(
                    "std::sort",
                    dist.name,
                    base,
                    universe_size,
                    std_sort,
                    1
                );
                std_row.speedup_vs_std_sort = 1.0;

                const SortFn native_sort = [universe_size](std::vector<int>& values) {
                    return dialsort_counting(values, universe_size);
                };

                const SortFn neural_sort = [&neural, universe_size](std::vector<int>& values) {
                    return neural.sort(values, universe_size);
                };

                const Row native_row = run_one(
                    "DialSort-Counting",
                    dist.name,
                    base,
                    universe_size,
                    native_sort,
                    std_row.best_ns
                );

                const Row neural_row = run_one(
                    "NeuralDialSort",
                    dist.name,
                    base,
                    universe_size,
                    neural_sort,
                    std_row.best_ns
                );

                print_row(native_row);
                print_row(neural_row);
                print_row(std_row);
                std::cout << "\n";
            }
        }
    }

    return 0;
}
