//
// Created by alex on 04.09.19.
//

#ifndef TRTMODEL_H
#define TRTMODEL_H


#include "basic_model.h"
#include <Utils.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;


class TRTModel: public BasicModel {
protected:
    float* imgCPU    = nullptr;
    float* imgCUDA  = nullptr;
private:
    size_t imgSize = 0;
    float3 scale;
    float3 shift;

public:
    TRTModel(
            const char* model_path,
            const char* input_blob,
            std::tuple<float, float, float>& scale,
            std::tuple<float, float, float>& shift,
            const std::vector<std::string>& output_blobs,
            uint32_t maxBatchSize);

    py::object Apply(py::array_t<uint8_t, py::array::c_style> image);

    ~TRTModel() override;
};

#endif //TRTMODEL_H
