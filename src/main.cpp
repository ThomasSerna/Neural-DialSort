#include "../include/dto/sort_result_dto.h"
#include "../include/dialsort.h"
#include "../include/neural_dialsort.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static constexpr int64_t TEST_N = 10000000;
static constexpr int64_t TEST_U = 65536;
static constexpr int64_t MIN_VALUE = 0;
static constexpr uint64_t SEED = 123;

std::vector<int64_t> generateRandomVector(int64_t n, int64_t u) {
    std::vector<int64_t> values(static_cast<size_t>(n));

    std::mt19937_64 rng(SEED);
    std::uniform_int_distribution<int64_t> dist(MIN_VALUE, MIN_VALUE + u - 1);

    for (int64_t i = 0; i < n; ++i) {
        values[static_cast<size_t>(i)] = dist(rng);
    }

    return values;
}

void printResult(const sort_result_dto& result) {
    std::cout << "\n=== Resultado: " << result.algorithmUsed << " ===\n";
    std::cout << "n               : " << result.n << "\n";
    std::cout << "u               : " << result.u << "\n";
    std::cout << "success         : " << (result.success ? "true" : "false") << "\n";
    std::cout << "tiempo          : "
              << std::fixed << std::setprecision(3)
              << result.elapsedMs << " ms\n";
}

int main() {
    try {
        std::vector<int64_t> input = generateRandomVector(TEST_N, TEST_U);

        dialsort nativeSorter(MIN_VALUE);
        neural_dialsort neuralSorter("model", MIN_VALUE, 8);

        sort_result_dto nativeResult = nativeSorter.sort(input, TEST_N, TEST_U);
        sort_result_dto neuralResult = neuralSorter.sort(input, TEST_N, TEST_U);

        printResult(nativeResult);
        printResult(neuralResult);

        const bool sameOutput = nativeResult.sortedVector == neuralResult.sortedVector;

        std::cout << "\n=== Comparacion ===\n";
        std::cout << "salidas iguales : " << (sameOutput ? "true" : "false") << "\n";

        if (nativeResult.elapsedMs > 0.0) {
            std::cout << "ONNX / nativo   : "
                      << std::fixed << std::setprecision(3)
                      << (neuralResult.elapsedMs / nativeResult.elapsedMs)
                      << "x\n";
        }

        return 0;

    } catch (const Ort::Exception& e) {
        std::cerr << "\nONNX Runtime error: " << e.what() << "\n";
        return 69;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 69;
    }
}
