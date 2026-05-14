cmake_minimum_required(VERSION 3.20)
project(neural_dialsort_variant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DSORT_ENABLE_ONNX "Build experimental ONNX-based Neural DialSort benchmark" OFF)

if(NOT DSORT_ENABLE_ONNX)
    message(STATUS "DSORT_ENABLE_ONNX=OFF: Neural DialSort sources and scripts are available, but the ONNX benchmark is not built.")
endif()

if(DSORT_ENABLE_ONNX)
    set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root path")
    set(DSORT_ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to the ONNX Runtime library")
    set(DSORT_ONNXRUNTIME_DLL "" CACHE FILEPATH "Path to onnxruntime.dll on Windows")

    if(NOT ONNXRUNTIME_DIR)
        message(FATAL_ERROR "Please pass -DONNXRUNTIME_DIR=/path/to/onnxruntime")
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

    add_executable(bench_neural_dialsort
            bench_neural_dialsort.cpp
            neural_dialsort/src/neural_dialsort.cpp)

    target_include_directories(bench_neural_dialsort PRIVATE
            neural_dialsort/include
            "${ONNXRUNTIME_DIR}/include")

    target_link_libraries(bench_neural_dialsort PRIVATE
            "${DSORT_ONNXRUNTIME_LIB}")

    target_compile_features(bench_neural_dialsort PRIVATE
            cxx_std_17)

    if(WIN32)
        if(NOT DSORT_ONNXRUNTIME_DLL AND EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
            set(DSORT_ONNXRUNTIME_DLL
                    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
                    CACHE FILEPATH
                    "Path to onnxruntime.dll on Windows"
                    FORCE)
        endif()

        if(DSORT_ONNXRUNTIME_DLL)
            add_custom_command(
                    TARGET bench_neural_dialsort POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DSORT_ONNXRUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:bench_neural_dialsort>")
        endif()
    endif()
endif()
