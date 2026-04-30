// benchmark_dialsort_onnx.cpp
//
// Benchmark C++ para un modelo ONNX de DialSort:
//   1. Genera un vector desordenado de N enteros dentro de [MIN_VALUE, MIN_VALUE + U - 1].
//   2. Ejecuta el modelo ONNX para producir el histograma H.
//   3. Proyecta H al vector finalmente ordenado.
//   4. Mide:
//        - tiempo de inferencia ONNX
//        - tiempo de proyección H -> vector ordenado
//        - tiempo total ONNX + proyección
//
// Requisitos:
//   - ONNX Runtime C++
//   - C++17
//
// En Windows recuerda que onnxruntime.dll debe estar junto al .exe o en el PATH.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ============================================================
// VARIABLES DE CONTROL
// Cambia manualmente estos valores para cada prueba.
// ============================================================

static const std::string MODEL_PATH = "model/dialsort_n100k_u10k.onnx";

// N debe coincidir con el --n usado al generar el modelo,
// a menos que hayas creado el ONNX con --dynamic-n.
static constexpr int64_t N = 100000;

// U debe coincidir con el --u usado al generar el modelo.
// Si U = 10000 y MIN_VALUE = 0, el rango válido es [0, 9999].
static constexpr int64_t U = 10000;

// Debe coincidir con el --min-value usado al generar el modelo.
static constexpr int64_t MIN_VALUE = 0;

// Corridas de benchmark. Las warmups no se reportan.
static constexpr int WARMUP_RUNS = 2;
static constexpr int MEASURE_RUNS = 10;

// Semilla para generar siempre el mismo vector desordenado.
static constexpr uint64_t SEED = 123;

// Modos soportados: "random", "sorted", "reverse", "skewed".
static const std::string INPUT_MODE = "random";

// Hilos internos de ONNX Runtime en CPU.
// Para medir de forma limpia puedes empezar con 1.
// Si quieres que ORT use más paralelismo, sube este valor.
static constexpr int ORT_INTRA_OP_THREADS = 1;

// Validación ligera: confirma que la salida está ordenada y conserva longitud.
// Esta validación es barata.
static constexpr bool VALIDATE_LIGHT = true;

// Validación estricta: compara contra std::sort(input).
// Para N = 10M puede tardar bastante, pero NO se incluye en el tiempo medido.
static constexpr bool VALIDATE_STRICT_WITH_STD_SORT = false;

// Imprimir primeros elementos para depurar.
static constexpr bool PRINT_SAMPLE = true;
static constexpr size_t SAMPLE_SIZE = 32;

// ============================================================

using Clock = std::chrono::high_resolution_clock;

struct RunTimes {
    double onnx_ms = 0.0;
    double projection_ms = 0.0;
    double total_ms = 0.0;
};

struct Stats {
    double min = 0.0;
    double mean = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string& s) {
    if (s.empty()) {
        return std::wstring();
    }

    int size_needed = MultiByteToWideChar(
        CP_UTF8,
        0,
        s.c_str(),
        static_cast<int>(s.size()),
        nullptr,
        0
    );

    if (size_needed <= 0) {
        throw std::runtime_error("No pude convertir MODEL_PATH de UTF-8 a UTF-16.");
    }

    std::wstring result(size_needed, L'\0');

    int converted = MultiByteToWideChar(
        CP_UTF8,
        0,
        s.c_str(),
        static_cast<int>(s.size()),
        result.data(),
        size_needed
    );

    if (converted <= 0) {
        throw std::runtime_error("Falló la conversión de MODEL_PATH de UTF-8 a UTF-16.");
    }

    return result;
}
#endif

double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Stats compute_stats(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("No hay valores para calcular estadísticas.");
    }

    Stats s;
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    double acc = 0.0;
    for (double x : values) {
        double d = x - s.mean;
        acc += d * d;
    }

    s.stddev = std::sqrt(acc / values.size());
    return s;
}

void print_stats_line(const std::string& name, const std::vector<double>& values) {
    Stats s = compute_stats(values);

    std::cout << std::left << std::setw(24) << name
              << "min=" << std::setw(10) << std::fixed << std::setprecision(3) << s.min
              << "mean=" << std::setw(10) << std::fixed << std::setprecision(3) << s.mean
              << "max=" << std::setw(10) << std::fixed << std::setprecision(3) << s.max
              << "std=" << std::setw(10) << std::fixed << std::setprecision(3) << s.stddev
              << "\n";
}

std::vector<int64_t> generate_input() {
    if (N <= 0) {
        throw std::runtime_error("N debe ser positivo.");
    }

    if (U <= 0) {
        throw std::runtime_error("U debe ser positivo.");
    }

    std::vector<int64_t> x(static_cast<size_t>(N));

    if (INPUT_MODE == "random") {
        std::mt19937_64 rng(SEED);
        std::uniform_int_distribution<int64_t> dist(MIN_VALUE, MIN_VALUE + U - 1);

        for (int64_t i = 0; i < N; ++i) {
            x[static_cast<size_t>(i)] = dist(rng);
        }
    } else if (INPUT_MODE == "sorted") {
        for (int64_t i = 0; i < N; ++i) {
            x[static_cast<size_t>(i)] = MIN_VALUE + (i % U);
        }
        std::sort(x.begin(), x.end());
    } else if (INPUT_MODE == "reverse") {
        for (int64_t i = 0; i < N; ++i) {
            x[static_cast<size_t>(i)] = MIN_VALUE + (i % U);
        }
        std::sort(x.begin(), x.end(), std::greater<int64_t>());
    } else if (INPUT_MODE == "skewed") {
        // 80% de los datos caen en el primer 5% del universo,
        // 20% se distribuyen sobre todo el universo.
        std::mt19937_64 rng(SEED);
        int64_t hot_u = std::max<int64_t>(1, U / 20);

        std::uniform_int_distribution<int64_t> hot_dist(MIN_VALUE, MIN_VALUE + hot_u - 1);
        std::uniform_int_distribution<int64_t> full_dist(MIN_VALUE, MIN_VALUE + U - 1);
        std::uniform_real_distribution<double> coin(0.0, 1.0);

        for (int64_t i = 0; i < N; ++i) {
            x[static_cast<size_t>(i)] = (coin(rng) < 0.80) ? hot_dist(rng) : full_dist(rng);
        }
    } else {
        throw std::runtime_error(
            "INPUT_MODE inválido. Usa: random, sorted, reverse o skewed."
        );
    }

    return x;
}

std::vector<int64_t> project_histogram_to_sorted_vector(
    const int64_t* histogram,
    int64_t histogram_size,
    int64_t expected_n
) {
    if (histogram == nullptr) {
        throw std::runtime_error("histogram es nullptr.");
    }

    if (histogram_size <= 0) {
        throw std::runtime_error("histogram_size debe ser positivo.");
    }

    std::vector<int64_t> sorted;
    sorted.resize(static_cast<size_t>(expected_n));

    int64_t pos = 0;

    for (int64_t i = 0; i < histogram_size; ++i) {
        int64_t count = histogram[i];

        if (count < 0) {
            throw std::runtime_error("El histograma tiene un conteo negativo.");
        }

        if (pos + count > expected_n) {
            throw std::runtime_error(
                "La suma del histograma excede N. Revisa N, U, MIN_VALUE o el modelo."
            );
        }

        int64_t value = MIN_VALUE + i;

        std::fill_n(
            sorted.begin() + static_cast<std::ptrdiff_t>(pos),
            static_cast<size_t>(count),
            value
        );

        pos += count;
    }

    if (pos != expected_n) {
        throw std::runtime_error(
            "La suma del histograma no coincide con N. "
            "Revisa que el input tenga la longitud esperada y el universo correcto."
        );
    }

    return sorted;
}

bool is_sorted_non_decreasing(const std::vector<int64_t>& x) {
    return std::is_sorted(x.begin(), x.end());
}

void print_sample(const std::string& label, const std::vector<int64_t>& x) {
    std::cout << label << " [";
    size_t limit = std::min(SAMPLE_SIZE, x.size());

    for (size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << x[i];
    }

    if (x.size() > limit) {
        std::cout << ", ... total=" << x.size();
    }

    std::cout << "]\n";
}

int64_t get_histogram_size_from_output(const Ort::Value& output) {
    Ort::TensorTypeAndShapeInfo info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();

    if (shape.size() != 1) {
        throw std::runtime_error("La salida del modelo debe ser un tensor 1D: histogram[U].");
    }

    if (shape[0] <= 0) {
        throw std::runtime_error("No pude inferir U desde la salida del modelo.");
    }

    return shape[0];
}

int main() {
    try {
        std::cout << "=== DialSort ONNX C++ Benchmark ===\n\n";
        std::cout << "MODEL_PATH       : " << MODEL_PATH << "\n";
        std::cout << "N                : " << N << "\n";
        std::cout << "U                : " << U << "\n";
        std::cout << "MIN_VALUE        : " << MIN_VALUE << "\n";
        std::cout << "Rango válido     : [" << MIN_VALUE << ", " << (MIN_VALUE + U - 1) << "]\n";
        std::cout << "INPUT_MODE       : " << INPUT_MODE << "\n";
        std::cout << "WARMUP_RUNS      : " << WARMUP_RUNS << "\n";
        std::cout << "MEASURE_RUNS     : " << MEASURE_RUNS << "\n";
        std::cout << "ORT threads      : " << ORT_INTRA_OP_THREADS << "\n\n";

        std::vector<int64_t> input = generate_input();

        if (PRINT_SAMPLE) {
            print_sample("Entrada", input);
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dialsort_onnx_benchmark");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(ORT_INTRA_OP_THREADS);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring model_path_w = utf8_to_wstring(MODEL_PATH);
        Ort::Session session(env, model_path_w.c_str(), session_options);
#else
        Ort::Session session(env, MODEL_PATH.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

        const char* input_name = input_name_alloc.get();
        const char* output_name = output_name_alloc.get();

        std::cout << "Input ONNX       : " << input_name << "\n";
        std::cout << "Output ONNX      : " << output_name << "\n\n";

        std::vector<int64_t> input_shape = {N};

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

        std::vector<const char*> input_names = {input_name};
        std::vector<const char*> output_names = {output_name};

        std::vector<double> onnx_times;
        std::vector<double> projection_times;
        std::vector<double> total_times;

        std::vector<int64_t> last_sorted_output;

        const int total_runs = WARMUP_RUNS + MEASURE_RUNS;

        for (int run = 0; run < total_runs; ++run) {
            bool is_warmup = run < WARMUP_RUNS;

            auto t0 = Clock::now();

            std::vector<Ort::Value> outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                1
            );

            auto t1 = Clock::now();

            if (outputs.empty()) {
                throw std::runtime_error("El modelo no produjo salidas.");
            }

            int64_t histogram_size = get_histogram_size_from_output(outputs[0]);

            if (histogram_size != U) {
                throw std::runtime_error(
                    "El U real de la salida no coincide con la variable U del benchmark. "
                    "Actualiza la variable U al inicio del archivo."
                );
            }

            const int64_t* histogram = outputs[0].GetTensorData<int64_t>();

            std::vector<int64_t> sorted_output =
                project_histogram_to_sorted_vector(histogram, histogram_size, N);

            auto t2 = Clock::now();

            double onnx_ms = elapsed_ms(t0, t1);
            double projection_ms = elapsed_ms(t1, t2);
            double total_ms = elapsed_ms(t0, t2);

            if (!is_warmup) {
                onnx_times.push_back(onnx_ms);
                projection_times.push_back(projection_ms);
                total_times.push_back(total_ms);
                last_sorted_output = std::move(sorted_output);

                std::cout << "run " << std::setw(2) << (run - WARMUP_RUNS + 1)
                          << " | onnx=" << std::setw(10) << std::fixed << std::setprecision(3) << onnx_ms << " ms"
                          << " | projection=" << std::setw(10) << std::fixed << std::setprecision(3) << projection_ms << " ms"
                          << " | total=" << std::setw(10) << std::fixed << std::setprecision(3) << total_ms << " ms"
                          << "\n";
            }
        }

        std::cout << "\n=== Estadisticas ms ===\n";
        print_stats_line("ONNX histogram", onnx_times);
        print_stats_line("H -> sorted vector", projection_times);
        print_stats_line("TOTAL", total_times);

        Stats total_stats = compute_stats(total_times);
        double best_total_seconds = total_stats.min / 1000.0;
        double throughput_mkeys_s = (static_cast<double>(N) / best_total_seconds) / 1e6;

        std::cout << "\nBest TOTAL throughput: "
                  << std::fixed << std::setprecision(3)
                  << throughput_mkeys_s << " M keys/s\n";

        if (VALIDATE_LIGHT) {
            bool ordered_ok = is_sorted_non_decreasing(last_sorted_output);
            bool length_ok = static_cast<int64_t>(last_sorted_output.size()) == N;

            std::cout << "\n=== Validacion ligera ===\n";
            std::cout << "Vector esta ordenado      : " << (ordered_ok ? "true" : "false") << "\n";
            std::cout << "Conserva longitud N       : " << (length_ok ? "true" : "false") << "\n";

            if (!ordered_ok || !length_ok) {
                throw std::runtime_error("Fallo la validacion ligera.");
            }
        }

        if (VALIDATE_STRICT_WITH_STD_SORT) {
            std::cout << "\nEjecutando validación estricta con std::sort fuera del benchmark...\n";

            std::vector<int64_t> expected = input;
            std::sort(expected.begin(), expected.end());

            bool strict_ok = expected == last_sorted_output;

            std::cout << "Salida coincide con std::sort: " << (strict_ok ? "true" : "false") << "\n";

            if (!strict_ok) {
                throw std::runtime_error("Falló la validación estricta contra std::sort.");
            }
        }

        if (PRINT_SAMPLE) {
            print_sample("Salida ordenada", last_sorted_output);
        }

        std::cout << "\nOK: benchmark terminado correctamente.\n";
        return EXIT_SUCCESS;

    } catch (const Ort::Exception& e) {
        std::cerr << "\nONNX Runtime error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
