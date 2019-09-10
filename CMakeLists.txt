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

file(GLOB cudaSources cuda/*.cu)
file(GLOB srcSources src/*.cpp src/*.cu)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda)
#include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pybind11)

cuda_add_library(OpenposeTensorRT SHARED ${cudaSources} ${srcSources})
#cuda_add_executable(OpenposeTensorRT ${cudaSources} ${srcSources} ${imagesSources})

#add_executable(OpenposeTensorRT ${cudaSources} ${srcSources} ${imagesSources})
target_link_libraries(OpenposeTensorRT
        nvinfer
        nvcaffe_parser
        nvinfer_plugin
        nvonnxparser
        protobuf
        pybind11::module
        ${PYTHON_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cupti_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_npp_LIBRARY}
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppi_LIBRARY}
        ${CUDA_npps_LIBRARY}
        ${CUDA_nvcuvenc_LIBRARY}
        ${CUDA_nvcuvid_LIBRARY}
        )

set_target_properties(OpenposeTensorRT PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")