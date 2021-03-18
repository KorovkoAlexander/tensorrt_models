//
// Created by alex on 04.09.19.
//

#include "TRTModel.h"
#include <cudaMappedMemory.h>
#include <pybind11/numpy.h>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"

TRTModel::TRTModel(
        const string& model_path,
        const uint8_t& device,
        const string& logs_path)
{

    if(!logs_path.empty()) {
        auto logger = spdlog::get("basic_logger");
        if (logger == nullptr)
            logger = spdlog::rotating_logger_mt("basic_logger", logs_path, 1048576 * 5, 1);
        spdlog::set_default_logger(logger);
    }
    spdlog::set_error_handler([](const std::string &msg) {});
    spdlog::flush_on(spdlog::level::info);

    if(CUDA_FAILED(setDevice(device))){
        throw runtime_error("Failed to properly set device");
    }

    bool res = LoadNetwork(
            model_path);
    if (!res)
        throw runtime_error("Failed to load the model");

    max_batch_size = engine->getMaxBatchSize();

    //apply test image for initialization
    py::array::ShapeContainer shape({
        static_cast<long>(max_batch_size),
        3, input_height, input_width
    });

    py::array::StridesContainer strides({
        static_cast<long>(max_batch_size)*input_width*input_height*3*8,
        input_width*input_height*3*8,
        input_width*input_height*8,
        input_width*8,
    }); //strides in bytes(like in regular numpy array)
    auto test_buffer = py::array_t<float, py::array::c_style>(shape, strides);
    auto ret = Apply(test_buffer);
}

py::object TRTModel::Apply(py::array_t<float, py::array::c_style> image)
{
    py::buffer_info image_info = image.request();

    if (image_info.ndim != 4)
        throw runtime_error("Number of dimensions nums be 4");

    if (input_width != image_info.shape[3]){
        spdlog::error("imgWidth must be equal to {}, but got input {}", input_width, image_info.shape[3]);
        throw runtime_error("imgWidth must be equal to " + to_string(input_width) +
                            ", but got input " + to_string(image_info.shape[3]));
    }

    if (input_height != image_info.shape[2]){
        spdlog::error("imgHeight must be equal to {}, but got input {}", input_height, image_info.shape[2]);
        throw runtime_error("imgHeight must be equal to " + to_string(input_height) +
                                ", but got input " + to_string(image_info.shape[2]));
    }

    if (image_info.shape[1] != 3){
        spdlog::error("Channels must be equal to 3, but got input {}", image_info.shape[1]);
        throw runtime_error("Channels must be equal to 3, but got input " + to_string(image_info.shape[1]));
    }

    if (max_batch_size < image_info.shape[0]){
        spdlog::error("Batch must be equal or less then {}, but got {}", max_batch_size, image_info.shape[0]);
        throw runtime_error("Batch must be equal or less then " + to_string(max_batch_size) +
                                ", but got " + to_string(image_info.shape[0]));
    }

    const int batchSize = image_info.shape[0];
    if (!context->allInputDimensionsSpecified())
        context->setBindingDimensions(0, Dims4{batchSize, 3, input_height, input_width});

    memcpy(input_tensor->host(), image_info.ptr,
           image_info.size * sizeof(float));

    // process with GIE
    vector<void*> inferenceBuffers = {input_tensor->device()};
    inferenceBuffers.reserve(output_tensors.size() + 1);
    for(const auto& x: output_tensors){
        inferenceBuffers.push_back(x.memory->device());
    }

    const bool result = context->executeV2(inferenceBuffers.data());

    if(!result)
    {
        spdlog::error(LOG_TRT "failed to execute tensorRT context");
        return py::none();
    }


    vector<py::array_t<float >> list_outputs;
    list_outputs.reserve(output_tensors.size());
    for(const auto& x: output_tensors){
        size_t mem_size = x.memory->get_size() * batchSize / max_batch_size;

        vector<int> shapes(x.dims.d, x.dims.d + x.dims.nbDims);
        if(shapes[0] < 0) // check batch size
            shapes[0] = batchSize;
        if(count_if(shapes.begin(), shapes.end(), [](const int& x){return x <= 0;}) > 0) {
            spdlog::error(LOG_TRT "One of output shape dimensions occurred to be less or equal to zero");
            for(const int& _x: shapes){
                spdlog::error(LOG_TRT "shape: {}", _x);
            }
            throw runtime_error("One of output shape dimensions occurred to be less or equal to zero");
        }
        py::array::ShapeContainer shape(shapes);

        py::array_t<float> out = py::array_t<float>(mem_size/sizeof(float));
        out.resize(shape);
        py::buffer_info out_info = out.request();
        memcpy(out_info.ptr, x.memory->host(), mem_size);

        list_outputs.push_back(move(out));
    }
    return py::cast(list_outputs);
}


PYBIND11_MODULE(tensorrt_models, m){
    py::enum_<precisionType >(m, "precisionType")
            .value("TYPE_FP32", TYPE_FP32 )
            .value("TYPE_FP16",  TYPE_FP16  )
            .value("TYPE_INT8",  TYPE_INT8  )
            .value("TYPE_FASTEST",  TYPE_FASTEST  )
            .export_values();
    py::enum_<deviceType>(m, "deviceType")
            .value("DEVICE_GPU", DEVICE_GPU )
            .value("DEVICE_DLA",  DEVICE_DLA  )
            .value("DEVICE_DLA_0",  DEVICE_DLA_0  )
            .value("DEVICE_DLA_1",  DEVICE_DLA_1  )
            .export_values();
    py::enum_<pixelFormat>(m, "pixelFormat")
            .value("RGB", RGB )
            .value("BGR",  BGR)
            .export_values();

    m.def("convertONNX", &convertONNX, "convert ONNX model into engine file",
            py::arg("modelFile"),
            py::arg("file_list") = "",
            py::arg("scale") = std::tuple<float, float, float>(58.395, 57.12 , 57.375),
            py::arg("shift") = std::tuple<float, float, float>(123.675, 116.28 , 103.53),
            py::arg("max_batch_size") = 1,
            py::arg("allowGPUFallback") = true,
            py::arg("device") = DEVICE_GPU,
            py::arg("precision") = TYPE_FP32,
            py::arg("format") = RGB,
            py::arg("logs_path") = "");

    py::class_<TRTModel>(m, "TRTModel")
            .def(py::init<const string&,
                    const uint8_t &, const string&>(),
                    py::arg("model_path"),
                    py::arg("device") = 0,
                    py::arg("logs_path") = "")
            .def("apply", &TRTModel::Apply)
            .def_property_readonly("input_dims", &TRTModel::getInputDims)
            .def_property_readonly("output_dims", &TRTModel::getOutputDims);
}
