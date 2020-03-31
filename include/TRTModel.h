//
// Created by alex on 04.09.19.
//

#ifndef TRTMODEL_H
#define TRTMODEL_H


#include "basic_model.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;


class TRTModel: public BasicModel {
private:
    cudaStream_t  mStream;
    size_t max_batch_size;

public:
    TRTModel(
            const string& model_path,
            const uint8_t& device,
            const string& logs_path);

    py::object Apply(py::array_t<float, py::array::c_style> image);

    ~TRTModel() override = default;
};

#endif //TRTMODEL_H
