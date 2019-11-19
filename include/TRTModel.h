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
            const string& model_path,
            const string& input_blob,
            const std::vector<std::string>& output_blobs,
            const std::tuple<float, float, float>& scale,
            const std::tuple<float, float, float>& shift,
            uint32_t maxBatchSize,
            const uint8_t& device,
            const string& logs_path);

    py::object Apply(py::array_t<uint8_t, py::array::c_style> image);

    ~TRTModel() override;
};

#endif //TRTMODEL_H
