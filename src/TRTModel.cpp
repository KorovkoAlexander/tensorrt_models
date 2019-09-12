//
// Created by alex on 04.09.19.
//

#include "TRTModel.h"
#include <cudaMappedMemory.h>

TRTModel::TRTModel(
        const char* model_path,
        const char* input_blob,
        const std::vector<std::string>& output_blobs,
        uint32_t maxBatchSize)
{
    bool res = LoadNetwork(model_path, input_blob, output_blobs, maxBatchSize);
    if (!res)
        throw runtime_error("Fuck! Failed to load the model :(");

    imgSize = DIMS_W(mInputDims) * DIMS_H(mInputDims) * sizeof(float) * 3;

    if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgSize) )
    {
        throw runtime_error("Fuck! Failed to allocate shared memory :(");
    }
}

TRTModel::~TRTModel() {
    if(!cudaDeallocMapped((void**)imgCPU)){
        cerr << "Cant deallocate cuda memory in dtor!" << endl;
    }

}

py::object TRTModel::Apply(py::array_t<uint8_t, py::array::c_style> image)
{
    py::buffer_info image_info = image.request();

    if (image_info.ndim != 3)
        throw runtime_error("Number of dimentions nums be 3");

    const int imgWidth = DIMS_W(mInputDims);
    if (imgWidth != image_info.shape[1]){
        cerr << "imgWidth must be equal to " << imgWidth << ", but got input is " <<  image_info.shape[1] << endl;
        return py::none();
    }

    const int imgHeight = DIMS_H(mInputDims);
    if (imgHeight != image_info.shape[0]){
        cerr << "imgHeight must be equal to " << imgHeight << ", but got input is " <<  image_info.shape[0] << endl;
        return py::none();
    }

    if(! loadImage((uint8_t *)image_info.ptr, (float3**)&imgCPU, imgWidth, imgHeight)){
        cerr << "Cant properly read numpy buffer :(" << endl;
        return py::none();
    }

    if( !imgCPU || imgWidth == 0 || imgHeight == 0)
    {
        cerr << "TRTModel::Apply( 0x" << imgCPU << ", "
             << imgWidth << ", " << imgHeight <<" ) -> invalid parameters" << endl;
        return py::none();
    }

    // downsample and convert to band-sequential BGR
    if( CUDA_FAILED(cudaPreImageNetRGB((float3*)imgCPU, imgWidth, imgHeight, mInputCUDA, mWidth, mHeight, GetStream())) )
    {
        printf("TRTModel::Apply() -- cudaPreImageNet failed\n");
        return py::none();
    }


    // process with GIE
//    void* inferenceBuffers[] = {mInputCUDA, mOutputs[OUTPUT_HEATMAP].CUDA, mOutputs[OUTPUT_PAF].CUDA};
    vector<void*> inferenceBuffers = {mInputCUDA};
    inferenceBuffers.reserve(mOutputs.size() + 1);
    for(const auto& x: mOutputs){
        inferenceBuffers.push_back(x.CUDA);
    }
    const bool result = mContext->enqueue(/*batchsize*/  1, inferenceBuffers.data(), GetStream(), nullptr);


    CUDA(cudaStreamSynchronize(mStream));


    if(!result)
    {
        printf("TRTModel::Apply() -- failed to execute tensorRT context\n");
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

    m.def("convertONNX", &convertONNX, "convert ONNX model into engine file",
            py::arg("modelFile"),
            py::arg("file_list"),
            py::arg("maxBatchSize"),
            py::arg("allowGPUFallback"),
            py::arg("device") = DEVICE_GPU,
            py::arg("precision") = TYPE_FP32);

    py::class_<TRTModel>(m, "TRTModel")
            .def(py::init([](
                    string model_path,
                    string input_blob,
                    vector<string>& output_blobs,
                    uint32_t max_batch_size
            ){
                return new TRTModel(model_path.c_str(), input_blob.c_str(), output_blobs, max_batch_size);
            }))
            .def("apply", &TRTModel::Apply);
}
