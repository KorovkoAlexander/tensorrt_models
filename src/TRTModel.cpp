//
// Created by alex on 04.09.19.
//

#include "TRTModel.h"
#include <cudaMappedMemory.h>

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

    mStream = cudaCreateStream(true);
    max_batch_size = engine->getMaxBatchSize();
}

TRTModel::~TRTModel(){
    if(CUDA_FAILED(cudaStreamDestroy(mStream))){
        spdlog::error("failed to destroy cuda stream");
    }
}

py::object TRTModel::Apply(py::array_t<float, py::array::c_style> image)
{
    py::buffer_info image_info = image.request();

    if (image_info.ndim != 4)
        throw runtime_error("Number of dimensions nums be 4");

    if (input_width != image_info.shape[3]){
        spdlog::error("imgWidth must be equal to {}, but got input {}", input_width, image_info.shape[3]);
        return py::none();
    }

    if (input_height != image_info.shape[2]){
        spdlog::error("imgHeight must be equal to {}, but got input {}", input_height, image_info.shape[2]);
        return py::none();
    }

    if (image_info.shape[1] != 3){
        spdlog::error("Channels must be equal to 3, but got input {}", image_info.shape[1]);
        return py::none();
    }

    if (max_batch_size < image_info.shape[0]){
        spdlog::error("Batch must be equal or less then {}, but got {}", max_batch_size, image_info.shape[0]);
        return py::none();
    }

    const int batchSize = image_info.shape[0];
    if (!context->allInputDimensionsSpecified())
        context->setBindingDimensions(0, Dims4{batchSize, 3, input_height, input_width});

    if(CUDA_FAILED(cudaMemcpy(
            input_tensor->ptr(),
            image_info.ptr,
            image_info.size * sizeof(float),
            cudaMemcpyHostToDevice))){
        spdlog::error(LOG_CUDA "Failed to copy memory to device!");
        return py::none();
    }

    // process with GIE
    vector<void*> inferenceBuffers = {input_tensor->ptr()};
    inferenceBuffers.reserve(output_tensors.size() + 1);
    for(const auto& x: output_tensors){
        inferenceBuffers.push_back(x.memory->ptr());
    }

    const bool result = context->enqueueV2(
            inferenceBuffers.data(),
            mStream,
            nullptr);

    CUDA(cudaStreamSynchronize(mStream));

    if(!result)
    {
        spdlog::error(LOG_TRT "failed to execute tensorRT context");
        return py::none();
    }


    vector<py::array_t<float >> list_outputs;
    list_outputs.reserve(output_tensors.size());
    for(const auto& x: output_tensors){
        size_t mem_size = x.memory->get_size() * batchSize / max_batch_size;
        py::array_t<float> out = py::array_t<float>(mem_size/sizeof(float));
        py::buffer_info out_info = out.request();
        if(CUDA_FAILED(cudaMemcpy(
                out_info.ptr,
                x.memory->ptr(),
                mem_size,
                cudaMemcpyDeviceToHost))){
            spdlog::error(LOG_CUDA "Failed to copy memory to device!");
            return py::none();
        }


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
