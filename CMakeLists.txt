cmake_minimum_required(VERSION 3.20)
project(neuralDialsort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "" CACHE PATH "Ruta raíz de ONNX Runtime")

if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Debes pasar -DONNXRUNTIME_DIR=/ruta/a/onnxruntime")
endif()

add_executable(neuralDialsort src/main.cpp
        src/neural_dialsort.cpp
        src/dialsort.cpp

        include/dto/sort_result_dto.h
        include/dialsort.h
        include/neural_dialsort.h
)

target_include_directories(
        neuralDialsort
        PRIVATE
        "${ONNXRUNTIME_DIR}/include"
)

target_link_libraries(
        neuralDialsort
        PRIVATE
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

add_custom_command(
        TARGET neuralDialsort POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:neuralDialsort>
)

add_custom_command(
        TARGET neuralDialsort POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/model"
        "$<TARGET_FILE_DIR:neuralDialsort>/model"
)
