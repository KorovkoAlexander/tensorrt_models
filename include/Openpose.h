//
// Created by alex on 04.09.19.
//

#ifndef OPENPOSETENSORRT_OPENPOSE_H
#define OPENPOSETENSORRT_OPENPOSE_H


#include "basic_model.h"
#include <OpenposeUtils.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;
#define OUTPUT_HEATMAP  1
#define OUTPUT_PAF 0



class Openpose: public BasicModel {
protected:
    float* imgCPU    = nullptr;
    float* imgCUDA  = nullptr;
private:
    const int imgWidth  = 1024;
    const int imgHeight = 576;
    const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 3;

public:
    Openpose(
            const char* model_path,
            const char* input_blob,
            const std::vector<std::string>& output_blobs,
            uint32_t maxBatchSize,
            precisionType type);

    py::object Apply(py::array_t<uint8_t, py::array::c_style> image);

    ~Openpose();
};

#endif //OPENPOSETENSORRT_OPENPOSE_H
