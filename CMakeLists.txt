cmake_minimum_required(VERSION 3.10)
project(tensorrt_models)

set(CMAKE_CXX_STANDARD 14)

if (UNIX)
    set(CMAKE_CXX_FLAGS "-fPIC -O3")
else()
    set(CMAKE_CXX_FLAGS "-O3")
endif (UNIX)

include_directories(/home/akorovko/Downloads/TensorRT-7.1.3.4/include)

set(TensorRT_LIBRARIES /home/akorovko/Downloads/TensorRT-7.1.3.4/lib)
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
add_subdirectory(spdlog)

cuda_add_library(tensorrt_models MODULE ${cudaSources} ${srcSources} ${imageSources})

target_link_libraries(tensorrt_models
        nvinfer
        nvinfer_plugin
        nvonnxparser
        pybind11::module
        spdlog::spdlog
        ${CUDA_LIBRARIES}
        )

set_target_properties(tensorrt_models PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")