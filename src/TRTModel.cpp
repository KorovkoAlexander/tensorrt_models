//
// Created by alex on 04.09.19.
//

#include "TRTModel.h"
#include <cudaMappedMemory.h>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"

TRTModel::TRTModel(
        const string& model_path,
        const string& input_blob,
        const std::vector<std::string>& output_blobs,
        const std::tuple<float, float, float>& _scale,
        const std::tuple<float, float, float>& _shift,
        uint32_t maxBatchSize,
        const uint8_t& device,
        const string& logs_path)
{

    if(!logs_path.empty()) {
        auto logger = spdlog::get("basic_logger");
        if (logger == nullptr)
            logger = spdlog::rotating_logger_mt("basic_logger", logs_path, 1048576 * 5, 1);
        spdlog::set_default_logger(logger);
        spdlog::set_error_handler([](const std::string &msg) {});
        spdlog::flush_on(spdlog::level::info);
    }

    if(CUDA_FAILED(setDevice(device))){
        throw runtime_error("Fuck! Failed to load the model :(");
    }

    bool res = LoadNetwork(
            model_path,
            input_blob,
            output_blobs,
            maxBatchSize);
    if (!res)
        throw runtime_error("Fuck! Failed to load the model :(");

    scale = make_float3(std::get<0>(_scale), std::get<1>(_scale), std::get<2>(_scale));
    shift = make_float3(std::get<0>(_shift), std::get<1>(_shift), std::get<2>(_shift));

    imgSize = maxBatchSize * DIMS_W(mInputDims) * DIMS_H(mInputDims) * sizeof(float) * 3;

    if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgSize) )
        throw runtime_error("Fuck! Failed to allocate zero-copy memory :(");
}

TRTModel::~TRTModel() {
    if(!cudaDeallocMapped((void**)imgCPU)){
        spdlog::error("Cant deallocate cuda memory in dtor!");
    }
    spdlog::drop_all();
}

py::object TRTModel::Apply(py::array_t<uint8_t, py::array::c_style> image)
{
    py::buffer_info image_info = image.request();

    if (image_info.ndim != 4)
        throw runtime_error("Number of dimentions nums be 4");

    const int imgWidth = DIMS_W(mInputDims);
    if (imgWidth != image_info.shape[2]){
        spdlog::error("imgWidth must be equal to {}, but got input is {}", imgWidth, image_info.shape[1]);
        return py::none();
    }

    const int imgHeight = DIMS_H(mInputDims);
    if (imgHeight != image_info.shape[1]){
        spdlog::error("imgHeight must be equal to {}, but got input is {}", imgHeight, image_info.shape[0]);
        return py::none();
    }

    const int batchSize = image_info.shape[0];

    if(! loadImage((uint8_t *)image_info.ptr, (float3**)&imgCPU, imgWidth, imgHeight, batchSize)){
        spdlog::error("Cant properly read numpy buffer :(");
        return py::none();
    }

    if( !imgCPU || imgWidth == 0 || imgHeight == 0)
    {
        spdlog::error("TRTModel::Apply( {}, {} ) -> invalid parameters", imgWidth, imgHeight);
        return py::none();
    }

    // downsample and convert to band-sequential BGR
    if( CUDA_FAILED(
            cudaPreImageNetScaleShiftRGB(
                    (float3*)imgCPU,
                    imgWidth,
                    imgHeight,
                    mInputCUDA,
                    mWidth,
                    mHeight,
                    batchSize,
                    scale,
                    shift,
                    GetStream())
                    ))
    {
        spdlog::error("TRTModel::Apply() -- cudaPreImageNet failed");
        return py::none();
    }

    // process with GIE
//    void* inferenceBuffers[] = {mInputCUDA, mOutputs[OUTPUT_HEATMAP].CUDA, mOutputs[OUTPUT_PAF].CUDA};
    vector<void*> inferenceBuffers = {mInputCUDA};
    inferenceBuffers.reserve(mOutputs.size() + 1);
    for(const auto& x: mOutputs){
        inferenceBuffers.push_back(x.CUDA);
    }
    const bool result = mContext->enqueue(/*batchsize*/  batchSize, inferenceBuffers.data(), GetStream(), nullptr);

    CUDA(cudaStreamSynchronize(mStream));

    if(!result)
    {
        spdlog::error("TRTModel::Apply() -- failed to execute tensorRT context");
        return py::none();
    }

    vector<py::array_t<float >> list_outputs;
    list_outputs.reserve(mOutputs.size());
    for(const auto& x: mOutputs){
        py::array_t<float> out = py::array_t<float>(x.size/sizeof(float));
        py::buffer_info out_info = out.request();
        memcpy(out_info.ptr, x.CPU, x.size);
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
            py::arg("scale") = std::tuple<float, float, float>(256, 256, 256),
            py::arg("shift") = std::tuple<float, float, float>(0.5, 0.5, 0.5),
            py::arg("maxBatchSize") = 1,
            py::arg("allowGPUFallback") = true,
            py::arg("device") = DEVICE_GPU,
            py::arg("precision") = TYPE_FP32,
            py::arg("format") = BGR,
            py::arg("logs_path") = "");

    py::class_<TRTModel>(m, "TRTModel")
            .def(py::init<const string&, const string&, const vector<string>&,
                    const std::tuple<float, float, float>& ,
                    const std::tuple<float, float, float>&,
                    uint32_t, const uint8_t &, const string&>(),
                    py::arg("model_path"),
                    py::arg("input_blob"),
                    py::arg("output_blobs"),
                    py::arg("scale") = std::tuple<float, float, float>(256, 256, 256),
                    py::arg("shift") = std::tuple<float, float, float>(0.5, 0.5, 0.5),
                    py::arg("max_batch_size") = 1,
                    py::arg("device") = 0,
                    py::arg("logs_path") = "")
            .def("apply", &TRTModel::Apply)
            .def_property_readonly("input_dims", &TRTModel::getInputDims)
            .def_property_readonly("output_dims", &TRTModel::getOutputDims);
}
