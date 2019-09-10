//
// Created by alex on 04.09.19.
//

#include "Openpose.h"
#include <cudaMappedMemory.h>

Openpose::Openpose(
        const char* model_path,
        const char* input_blob,
        const std::vector<std::string>& output_blobs,
        uint32_t maxBatchSize,
        precisionType type)
{
    bool res = LoadNetwork(model_path, input_blob, output_blobs, maxBatchSize, type);
    if (!res)
        throw runtime_error("Fuck! Failed to load the model :(");

    if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgSize) )
    {
        throw runtime_error("Fuck! Failed to allocate shared memory :(");
    }
}

Openpose::~Openpose() {
    if(!cudaDeallocMapped((void**)imgCPU)){
        cerr << "Cant deallocate cuda memory in dtor!" << endl;
    }

}

py::object Openpose::Apply(py::array_t<uint8_t, py::array::c_style> image)
{
    py::buffer_info image_info = image.request();

    if (image_info.ndim != 3)
        throw runtime_error("Number of dimentions nums be 3");

    if (imgWidth != image_info.shape[1]){
        cerr << "imgWidth must be equal to " << imgWidth << ", but got input is " <<  image_info.shape[1] << endl;
        return py::none();
    }

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
        cerr << "Openpose::Apply( 0x" << imgCPU << ", "
             << imgWidth << ", " << imgHeight <<" ) -> invalid parameters" << endl;
        return py::none();
    }

    // downsample and convert to band-sequential BGR
    if( CUDA_FAILED(cudaPreImageNetRGB((float3*)imgCPU, imgWidth, imgHeight, mInputCUDA, mWidth, mHeight, GetStream())) )
    {
        printf("Openpose::Apply() -- cudaPreImageNet failed\n");
        return py::none();
    }


    // process with GIE
    void* inferenceBuffers[] = {mInputCUDA, mOutputs[OUTPUT_HEATMAP].CUDA, mOutputs[OUTPUT_PAF].CUDA};
    const bool result = mContext->enqueue(/*batchsize*/  1, inferenceBuffers, GetStream(), nullptr);


    CUDA(cudaStreamSynchronize(mStream));


    if(!result)
    {
        printf("Openpose::Apply() -- failed to execute tensorRT context\n");
        return py::none();
    }

    py::array_t<float> heatmaps = py::array_t<float>(mOutputs[OUTPUT_HEATMAP].size/sizeof(float));
    py::array_t<float> paf = py::array_t<float>(mOutputs[OUTPUT_PAF].size/sizeof(float));

    py::buffer_info heatmap_info = heatmaps.request();
    py::buffer_info paf_info = paf.request();


    memcpy(heatmap_info.ptr, mOutputs[OUTPUT_HEATMAP].CPU, mOutputs[OUTPUT_HEATMAP].size);
    memcpy(paf_info.ptr, mOutputs[OUTPUT_PAF].CPU, mOutputs[OUTPUT_PAF].size);

    return py::make_tuple(heatmaps, paf);
}

PYBIND11_MODULE(OpenposeTensorRT, m){
    py::enum_<precisionType>(m, "precisionType")
            .value("TYPE_INT8", precisionType::TYPE_INT8)
            .value("TYPE_FP16", precisionType::TYPE_FP16)
            .value("TYPE_FP32", precisionType::TYPE_FP32)
            .value("TYPE_FASTEST", precisionType::TYPE_FASTEST)
            .value("TYPE_DISABLED", precisionType::TYPE_DISABLED)
            .export_values();

    py::class_<Openpose>(m, "Openpose")
            .def(py::init([](
                    string model_path,
                    string input_blob,
                    vector<string>& output_blobs,
                    uint32_t max_batch_size,
                    precisionType type
            ){
                return new Openpose(model_path.c_str(), input_blob.c_str(), output_blobs, max_batch_size, type);
            }))
            .def("apply", &Openpose::Apply);
}
