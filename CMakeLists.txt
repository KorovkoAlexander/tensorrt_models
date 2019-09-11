cmake_minimum_required(VERSION 3.14)
project(OpenposeTensorRT)

set(CMAKE_CXX_STANDARD 11)

include_directories(/home/alex/Downloads/TensorRT-5.1.2.2/include)

set(TensorRT_LIBRARIES /home/alex/Downloads/TensorRT-5.1.2.2/lib)
link_directories(${TensorRT_LIBRARIES})

find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB cudaSources cuda/*.cu src/*.cu)
file(GLOB srcSources src/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda)

add_subdirectory(pybind11)

cuda_add_library(OpenposeTensorRT MODULE ${cudaSources} ${srcSources})

target_link_libraries(OpenposeTensorRT
        nvinfer
        nvinfer_plugin
        nvonnxparser
        pybind11::module
        ${CUDA_LIBRARIES}
        )

set_target_properties(OpenposeTensorRT PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")