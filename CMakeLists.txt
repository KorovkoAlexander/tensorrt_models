cmake_minimum_required(VERSION 3.14)
project(tensorrt_models)

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
file(GLOB imageSources image/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda)
include_directories(${PROJECT_SOURCE_DIR}/image)
include_directories(${PROJECT_SOURCE_DIR}/image/stb)

add_subdirectory(pybind11)

cuda_add_library(tensorrt_models MODULE ${cudaSources} ${srcSources} ${imageSources})

target_link_libraries(tensorrt_models
        nvinfer
        nvinfer_plugin
        nvonnxparser
        pybind11::module
        ${CUDA_LIBRARIES}
        )

set_target_properties(tensorrt_models PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")