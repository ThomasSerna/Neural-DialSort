cmake_minimum_required(VERSION 3.27)
project(dsortcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Original DialSort targets
# ---------------------------------------------------------------------------

add_executable(dsortcpp
        dialsort_benchmark_v2.cpp)

add_executable(bench_counting
        bench_vs_dialsort_vs_counting_bench.cpp
        Dialsortvsskasort.cpp
        Dialsortvsips4oparallel.cpp)

add_executable(bench_ips4o
        DialsortvsIps4o.cpp)

target_include_directories(bench_ips4o PRIVATE
        ${CMAKE_SOURCE_DIR}/ips4o/include)

target_compile_definitions(bench_ips4o PRIVATE
        SKIP_IPS4O)

# ---------------------------------------------------------------------------
# Optional ONNX-Accelerated-DialSort benchmark
# ---------------------------------------------------------------------------

option(DSORT_ENABLE_ONNX "Build experimental ONNX-Accelerated-DialSort benchmark" OFF)

if(NOT DSORT_ENABLE_ONNX)
    message(STATUS "DSORT_ENABLE_ONNX=OFF: ONNX-Accelerated-DialSort sources and scripts are available, but the ONNX benchmark is not built.")
endif()

if(DSORT_ENABLE_ONNX)
    set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root path")
    set(DSORT_ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to the ONNX Runtime library")
    set(DSORT_ONNXRUNTIME_DLL "" CACHE FILEPATH "Path to onnxruntime.dll on Windows")

    if(NOT ONNXRUNTIME_DIR)
        message(FATAL_ERROR "Please pass -DONNXRUNTIME_DIR=/path/to/onnxruntime")
    endif()

    if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers were not found at: ${ONNXRUNTIME_DIR}/include")
    endif()

    if(NOT DSORT_ONNXRUNTIME_LIB)
        if(WIN32)
            set(_dsort_onnxruntime_default_lib "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
        elseif(APPLE)
            set(_dsort_onnxruntime_default_lib "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
        else()
            set(_dsort_onnxruntime_default_lib "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
        endif()

        set(DSORT_ONNXRUNTIME_LIB
                "${_dsort_onnxruntime_default_lib}"
                CACHE FILEPATH
                "Path to the ONNX Runtime library"
                FORCE)
    endif()

    if(NOT EXISTS "${DSORT_ONNXRUNTIME_LIB}")
        message(FATAL_ERROR "ONNX Runtime library was not found at: ${DSORT_ONNXRUNTIME_LIB}")
    endif()

    add_executable(bench_onnx_accelerated_dialsort
            bench_onnx_accelerated_dialsort.cpp
            onnx_accelerated_dialsort/src/onnx_accelerated_dialsort.cpp)

    target_include_directories(bench_onnx_accelerated_dialsort PRIVATE
            onnx_accelerated_dialsort/include
            "${ONNXRUNTIME_DIR}/include")

    target_link_libraries(bench_onnx_accelerated_dialsort PRIVATE
            "${DSORT_ONNXRUNTIME_LIB}")

    target_compile_features(bench_onnx_accelerated_dialsort PRIVATE
            cxx_std_17)

    # On Windows, copy the ONNX Runtime DLL next to the benchmark executable so
    # the benchmark does not accidentally load another onnxruntime.dll from PATH.
    if(WIN32)
        if(NOT DSORT_ONNXRUNTIME_DLL)
            if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
                set(DSORT_ONNXRUNTIME_DLL
                        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
                        CACHE FILEPATH
                        "Path to onnxruntime.dll on Windows"
                        FORCE)
            elseif(EXISTS "${ONNXRUNTIME_DIR}/bin/onnxruntime.dll")
                set(DSORT_ONNXRUNTIME_DLL
                        "${ONNXRUNTIME_DIR}/bin/onnxruntime.dll"
                        CACHE FILEPATH
                        "Path to onnxruntime.dll on Windows"
                        FORCE)
            endif()
        endif()

        if(DSORT_ONNXRUNTIME_DLL AND EXISTS "${DSORT_ONNXRUNTIME_DLL}")
            add_custom_command(
                    TARGET bench_onnx_accelerated_dialsort POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DSORT_ONNXRUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:bench_onnx_accelerated_dialsort>")
        else()
            message(WARNING "onnxruntime.dll was not found. The benchmark may load another DLL from PATH.")
        endif()
    endif()
endif()
