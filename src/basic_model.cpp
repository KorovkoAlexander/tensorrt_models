//
// Created by alex on 04.09.19.
//

#include "basic_model.h"
#include "cudaMappedMemory.h"

#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <iostream>
#include <fstream>
#include <map>

#define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
#define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime

class Logger : public nvinfer1::ILogger
{
    void log( Severity severity, const char* msg ) override
    {
        if( severity != Severity::kINFO /*|| mEnableDebug*/ )
                spdlog::info(LOG_TRT "{}", msg);
    }
} gLogger;

static inline bool isFp16Enabled( nvinfer1::IBuilder* builder )
{
#if NV_TENSORRT_MAJOR < 4
    return builder->getHalf2Mode();
#else
    return builder->getFp16Mode();
#endif
}

static inline bool isInt8Enabled( nvinfer1::IBuilder* builder )
{
#if NV_TENSORRT_MAJOR >= 4
    return builder->getInt8Mode();
#else
    return false;
#endif
}

#if NV_TENSORRT_MAJOR >= 4
static inline const char* dataTypeToStr( nvinfer1::DataType type )
{
    switch(type)
    {
        case nvinfer1::DataType::kFLOAT:	return "FP32";
        case nvinfer1::DataType::kHALF:	return "FP16";
        case nvinfer1::DataType::kINT8:	return "INT8";
        case nvinfer1::DataType::kINT32:	return "INT32";
    }

}

static inline const char* dimensionTypeToStr( nvinfer1::DimensionType type )
{
    switch(type)
    {
        case nvinfer1::DimensionType::kSPATIAL:	 return "SPATIAL";
        case nvinfer1::DimensionType::kCHANNEL:	 return "CHANNEL";
        case nvinfer1::DimensionType::kINDEX:	 return "INDEX";
        case nvinfer1::DimensionType::kSEQUENCE: return "SEQUENCE";
    }

}
#endif

#if NV_TENSORRT_MAJOR > 1
static inline nvinfer1::Dims validateDims( const nvinfer1::Dims& dims )
{
    if( dims.nbDims == nvinfer1::Dims::MAX_DIMS )
        return dims;

    nvinfer1::Dims dims_out = dims;

    // TRT doesn't set the higher dims, so make sure they are 1
    for( int n=dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
        dims_out.d[n] = 1;

    return dims_out;
}
#endif

bool loadImage(uint8_t * img, float3** cpu, const int& imgWidth, const int& imgHeight, const int& batchSize){
    if( !img )
        return false;

    const int imgChannels = 3;

//    float4 mean  = make_float4(0,0,0,0);

    if (cpu == nullptr)
    {
        std::cerr << "Cant load the image! No cpu memory provided.." << std::endl;
        return false;
    }

    float3* cpuPtr = *cpu;

    for (int b = 0; b < batchSize; b++){
        const size_t bOffset = b*imgWidth*imgHeight*imgChannels* sizeof(uint8_t);

        for( int y=0; y < imgHeight; y++ )
        {
            const size_t yOffset = bOffset + y * imgWidth * imgChannels * sizeof(uint8_t);

            for( int x=0; x < imgWidth; x++ )
            {
                #define GET_PIXEL(channel)	    float(img[offset + channel])
                #define SET_PIXEL_FLOAT3(_r,_g,_b) cpuPtr[b*imgWidth*imgHeight + y*imgWidth + x] = make_float3(_r,_g,_b)

                const size_t offset = yOffset + x * imgChannels * sizeof(uint8_t);
                SET_PIXEL_FLOAT3(GET_PIXEL(0), GET_PIXEL(1), GET_PIXEL(2));

            }
        }
    }

//    for( int y=0; y < imgHeight; y++ )
//    {
//        const size_t yOffset = y * imgWidth * imgChannels * sizeof(uint8_t);
//
//        for( int x=0; x < imgWidth; x++ )
//        {
//            #define GET_PIXEL(channel)	    float(img[offset + channel])
//            #define SET_PIXEL_FLOAT3(r,g,b) cpuPtr[y*imgWidth+x] = make_float3(r,g,b)
//
//            const size_t offset = yOffset + x * imgChannels * sizeof(uint8_t);
//            SET_PIXEL_FLOAT3(GET_PIXEL(0), GET_PIXEL(1), GET_PIXEL(2));
//
//        }
//    }
    return true;
}


#if NV_TENSORRT_MAJOR >= 5
static inline nvinfer1::DeviceType deviceTypeToTRT( deviceType type )
{
    switch(type)
    {
        case DEVICE_GPU:	return nvinfer1::DeviceType::kGPU;
            //case DEVICE_DLA:	return nvinfer1::DeviceType::kDLA;
#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0
        case DEVICE_DLA_0:	return nvinfer1::DeviceType::kDLA0;
		case DEVICE_DLA_1:	return nvinfer1::DeviceType::kDLA1;
#else
        case DEVICE_DLA_0:	return nvinfer1::DeviceType::kDLA;
        case DEVICE_DLA_1:	return nvinfer1::DeviceType::kDLA;
#endif
    }
}
#endif

std::vector<std::map<std::string, uint32_t >> BasicModel::getOutputDims() const{
    std::vector<std::map<std::string, uint32_t >> out;
    out.reserve(mOutputs.size());
    for(const auto& x: mOutputs){
        out.push_back(
                {
                        {"width", DIMS_W(x.dims)},
                        {"height", DIMS_H(x.dims)},
                        {"channels", DIMS_C(x.dims)}
                });
    }
    return out;
}

//---------------------------------------------------------------------

// constructor
BasicModel::BasicModel()
{
    mEngine  = nullptr;
    mInfer   = nullptr;
    mContext = nullptr;
    mStream  = nullptr;

    mWidth          = 0;
    mHeight         = 0;
    mInputSize      = 0;
    mMaxBatchSize   = 0;
    mInputCPU       = nullptr;
    mInputCUDA      = nullptr;

    mDevice    	   = DEVICE_GPU;
    mAllowGPUFallback = false;
}


// Destructor
BasicModel::~BasicModel(){
    mContext->destroy();
    mEngine->destroy();
    mInfer->destroy();


    if(!cudaDeallocMapped((void**)mInputCPU)){
        spdlog::error("Cant deallocate mInputCPU in dtor!");
    }
    for(auto& x : mOutputs){
        if(!cudaDeallocMapped((void**)x.CPU)){
            spdlog::error("Cant deallocate mOutputs in dtor!");
        }
    }
    if(CUDA_FAILED(cudaDeviceReset())){
        spdlog::error("Cant reset the device in dtor!");
    }

}


// LoadNetwork
bool BasicModel::LoadNetwork(const std::string& model_path,
                             const std::string& input_blob,
                             const std::string& output_blob,
                             uint32_t maxBatchSize,
                             deviceType device,
                             bool allowGPUFallback,
                             cudaStream_t stream)
{
    std::vector<std::string> outputs;
    outputs.emplace_back(output_blob);

    return LoadNetwork(model_path, input_blob, outputs, maxBatchSize, device, allowGPUFallback, stream);
}


// LoadNetwork
bool BasicModel::LoadNetwork(const std::string& model_path_,
                             const std::string& input_blob,
                             const std::vector<std::string>& output_blobs,
                             uint32_t maxBatchSize,
                             deviceType device, bool allowGPUFallback,
                             cudaStream_t stream)
{
    return LoadNetwork(model_path_,
                       input_blob,
                       Dims3(1,1,1),
                       output_blobs,
                       maxBatchSize,
                       device,
                       allowGPUFallback,
                       stream);
}


// LoadNetwork
bool BasicModel::LoadNetwork(
        const std::string& model_path,
        const std::string& input_blob,
        const Dims3& input_dims,
        const std::vector<std::string>& output_blobs,
        uint32_t maxBatchSize,
        deviceType device,
        bool allowGPUFallback,
        cudaStream_t stream
)
{
    spdlog::info("TensorRT version {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    /*
     * load NV inference plugins
     */
    static bool loadedPlugins = false;

    if( !loadedPlugins )
    {
        spdlog::info(LOG_TRT "loading NVIDIA plugins...");

        loadedPlugins = initLibNvInferPlugins(&gLogger, "");

        if( !loadedPlugins )
            spdlog::error(LOG_TRT "failed to load NVIDIA plugins");
        else
            spdlog::info(LOG_TRT "completed loading NVIDIA plugins");
    }

    /*
     * attempt to load network from cache before profiling with tensorRT
     */

    mCacheEnginePath = model_path;
    spdlog::info(LOG_TRT "attempting to open engine cache file {}", mCacheEnginePath.c_str());

    std::ifstream cache(model_path, std::ios::binary);

    if( !cache )
    {
        spdlog::error(LOG_TRT "cache file not found, profiling network model on device");
        return false;
    }


    /*
     * create runtime inference engine execution context
     */
    nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);

    if( !infer )
    {
        spdlog::error(LOG_TRT "failed to create InferRuntime");
        return false;
    }


    // if using DLA, set the desired core before deserialization occurs
    if( device == DEVICE_DLA_0 )
    {
        spdlog::info(LOG_TRT "enabling DLA core 0");
        infer->setDLACore(0);
    }
    else if( device == DEVICE_DLA_1 )
    {
        spdlog::info(LOG_TRT "enabling DLA core 1");
        infer->setDLACore(1);
    }

    cache.seekg(0, cache.end);
    const int length = cache.tellg();
    cache.seekg(0, cache.beg);

    void* modelMem = malloc(length);

    if( !modelMem )
    {
        spdlog::error(LOG_TRT "failed to allocate {} bytes to deserialize model", length);
        return false;
    }

    cache.read((char*)modelMem, length);
    cache.close();
    spdlog::info(LOG_TRT "device {} loaded", model_path.c_str());

    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, length, nullptr);
    free(modelMem);

    if( !engine )
    {
        spdlog::error(LOG_TRT "failed to create CUDA engine");
        return false;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    if( !context )
    {
        spdlog::error(LOG_TRT "failed to create execution context");
        return false;
    }

    spdlog::info(LOG_TRT "CUDA engine context initialized with {} bindings", engine->getNbBindings());

    mInfer   = infer;
    mEngine  = engine;
    mContext = context;

    SetStream(stream);	// set default device stream
    /*
     * print out binding info
     */
    const int numBindings = engine->getNbBindings();

    for( int n=0; n < numBindings; n++ )
    {
        spdlog::info(LOG_TRT "binding -- index   {}", n);

        const char* bind_name = engine->getBindingName(n);

        spdlog::info("               -- name    '{}'", bind_name);
        spdlog::info("               -- type    {}", dataTypeToStr(engine->getBindingDataType(n)));
        spdlog::info("               -- in/out  {}", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

        const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

        spdlog::info("               -- # dims  {}", bind_dims.nbDims);

        for( int i=0; i < bind_dims.nbDims; i++ )
            spdlog::info("               -- dim #{}  {} ({})\n", i, bind_dims.d[i],
                    dimensionTypeToStr(bind_dims.type[i]));
    }

    /*
     * determine dimensions of network input bindings
     */
    const int inputIndex = engine->getBindingIndex(input_blob.c_str());

    spdlog::info(LOG_TRT "binding to input 0 %s  binding index:  {}", input_blob.c_str(), inputIndex);

    nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));

    size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
    spdlog::info(LOG_TRT "binding to input 0 {}  dims (b={} c={} h={} w={}) size={}",
           input_blob.c_str(),
           maxBatchSize,
           DIMS_C(inputDims),
           DIMS_H(inputDims),
           DIMS_W(inputDims),
           inputSize);


    /*
     * allocate memory to hold the input buffer
     */
    if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
    {
        spdlog::error(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, {} bytes", inputSize);
        return false;
    }

    mInputSize    = inputSize;
    mWidth        = DIMS_W(inputDims);
    mHeight       = DIMS_H(inputDims);
    mMaxBatchSize = maxBatchSize;


    /*
     * setup network output buffers
     */
    const int numOutputs = output_blobs.size();

    for( int n=0; n < numOutputs; n++ )
    {
        const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
        spdlog::info(LOG_TRT "binding to output {} {}  binding index:  {}", n, output_blobs[n].c_str(), outputIndex);

        nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));

        size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
        spdlog::info(LOG_TRT "binding to output {} {}  dims (b={} c={} h={} w={}) size={}",
                n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims),
                DIMS_H(outputDims), DIMS_W(outputDims), outputSize);

        // allocate output memory
        void* outputCPU  = nullptr;
        void* outputCUDA = nullptr;

        //if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
        if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
        {
            spdlog::error(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, {} bytes", outputSize);
            return false;
        }

        outputLayer l;

        l.CPU  = (float*)outputCPU;
        l.CUDA = (float*)outputCUDA;
        l.size = outputSize;

        DIMS_W(l.dims) = DIMS_W(outputDims);
        DIMS_H(l.dims) = DIMS_H(outputDims);
        DIMS_C(l.dims) = DIMS_C(outputDims);

        l.name = output_blobs[n];
        mOutputs.push_back(l);
    }


    DIMS_W(mInputDims) = DIMS_W(inputDims);
    DIMS_H(mInputDims) = DIMS_H(inputDims);
    DIMS_C(mInputDims) = DIMS_C(inputDims);

    mModelPath        = model_path;
    mInputBlobName    = input_blob;
    mDevice           = device;
    mAllowGPUFallback = allowGPUFallback;


    spdlog::info("{} initialized.\n", mModelPath.c_str());
    return true;
}


// CreateStream
cudaStream_t BasicModel::CreateStream( bool nonBlocking )
{
    uint32_t flags = cudaStreamDefault;

    if( nonBlocking )
        flags = cudaStreamNonBlocking;

    cudaStream_t stream = nullptr;

    if( CUDA_FAILED(cudaStreamCreateWithFlags(&stream, flags)) )
        return nullptr;

    SetStream(stream);
    return stream;
}


// SetStream
void BasicModel::SetStream( cudaStream_t stream )
{
    mStream = stream;

    if( !mStream )
        return;
}

cudaError_t BasicModel::setDevice(int device) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if(error != cudaSuccess){
        spdlog::error(LOG_CUDA "Failed to get device count");
        return error;
    }

    if(device >= device_count) {
        spdlog::error(LOG_CUDA "Selected device num is greater than max number of devices");
        return cudaErrorDevicesUnavailable;
    }

    error = cudaSetDevice(device);
    if(error != cudaSuccess){
        spdlog::error(LOG_CUDA "Failed to set GPU to {%u}", device);
        return error;
    }
}

const char* precisionTypeToStr( precisionType type )
{
    switch(type)
    {
        case TYPE_DISABLED:	return "DISABLED";
        case TYPE_FASTEST:	return "FASTEST";
        case TYPE_FP32:	return "FP32";
        case TYPE_FP16:	return "FP16";
        case TYPE_INT8:	return "INT8";
    }
}

bool convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 const std::tuple<float, float, float>& scale,
                 const std::tuple<float, float, float>& shift,
                 unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 const deviceType& device,
                 precisionType precision,
                 const pixelFormat& format,
                 const std::string& logs_path)
{
    spdlog::flush_on(spdlog::level::info);
    if(!logs_path.empty()) {
        auto logger = spdlog::get("convertion_logger");
        if (logger == nullptr)
            logger = spdlog::rotating_logger_mt("convertion_logger", logs_path, 1048576 * 5, 1);
        spdlog::set_default_logger(logger);
        spdlog::set_error_handler([](const std::string &msg) {});
    }
    // create API root class - must span the lifetime of the engine usage
    nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    builder->setAverageFindIterations(2);

    spdlog::info(LOG_TRT "loading {}", modelFile.c_str());


    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if( !parser )
    {
        spdlog::error(LOG_TRT "failed to create nvonnxparser::IParser instance");
        network->destroy();
        builder->destroy();
        if(CUDA_FAILED(cudaDeviceReset())){
            spdlog::error("Cant reset the device !");
        }
        spdlog::drop_all();
        return false;
    }

    if( !parser->parseFromFile(modelFile.c_str(), (int)nvinfer1::ILogger::Severity::kERROR) )
    {
        spdlog::error(LOG_TRT "failed to parse ONNX model '{}'", modelFile.c_str());
        parser->destroy();
        network->destroy();
        builder->destroy();
        if(CUDA_FAILED(cudaDeviceReset())){
            spdlog::error("Cant reset the device !");
        }
        spdlog::drop_all();
        return false;
    }

    // extract the dimensions of the network input blobs
    nvinfer1::Dims3 inputDimensions = static_cast<nvinfer1::Dims3&&>(network->getInput(0)->getDimensions());


    // build the engine
    spdlog::info(LOG_TRT "configuring CUDA engine");

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);


    // set up the builder for the desired precision
    std::vector<precisionType > precisions = {TYPE_FP32};
    if(builder->platformHasFastFp16()){
        precisions.push_back(TYPE_FP16);
    }
    if(builder->platformHasFastInt8() && !file_list.empty()) {
        precisions.push_back(TYPE_INT8);
    }

    precisionType best_precision = precisions[precisions.size() -1];

    if(precision == TYPE_FASTEST){
        precision = best_precision;
    } else if (precision >= best_precision){
        precision = best_precision;
    }

    spdlog::info(LOG_CUDA "using precision {}", precisionTypeToStr(precision));

    if( precision == TYPE_INT8)
    {
        builder->setInt8Mode(true);

        int imgWidth = DIMS_W(inputDimensions);
        int imgHeight = DIMS_H(inputDimensions);
        int imgChannels = DIMS_C(inputDimensions);



        nvinfer1::IInt8Calibrator* calibrator= new EntropyCalibrator(
                file_list,
                imgWidth,
                imgHeight,
                imgChannels,
                make_float3(std::get<0>(scale), std::get<1>(scale), std::get<2>(scale)),
                make_float3(std::get<0>(shift), std::get<1>(shift), std::get<2>(shift))
                );

        builder->setInt8Calibrator(calibrator);
    }
    else if( precision == TYPE_FP16 )
    {
        builder->setFp16Mode(true);
    }


    // set the default device type
    builder->setDefaultDeviceType(deviceTypeToTRT(device));

    if( allowGPUFallback )
        builder->allowGPUFallback(true);

#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
    if( device == DEVICE_DLA_0 )
        builder->setDLACore(0);
    else if( device == DEVICE_DLA_1 )
        builder->setDLACore(1);
#endif


    // build CUDA engine
    spdlog::info(LOG_TRT "building FP16:  {}", isFp16Enabled(builder) ? "ON" : "OFF");
    spdlog::info(LOG_TRT "building INT8:  {}", isInt8Enabled(builder) ? "ON" : "OFF");
    spdlog::info(LOG_TRT "building CUDA engine (this may take a few minutes the first time a network is loaded)");

    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

    if( !engine )
    {
        spdlog::error(LOG_TRT "failed to build CUDA engine");
        parser->destroy();
        network->destroy();
        builder->destroy();
        if(CUDA_FAILED(cudaDeviceReset())){
            spdlog::error("Cant reset the device !");
        }
        spdlog::drop_all();
        return false;
    }

    spdlog::info(LOG_TRT "completed building CUDA engine");

    // we don't need the network definition any more, and we can destroy the parser
    network->destroy();

    nvinfer1::IHostMemory* serMem = engine->serialize();

    if( !serMem )
    {
        spdlog::error(LOG_TRT "failed to serialize CUDA engine");
        parser->destroy();
        engine->destroy();
        builder->destroy();
        if(CUDA_FAILED(cudaDeviceReset())){
            spdlog::error("Cant reset the device !");
        }
        spdlog::drop_all();
        return false;
    }


    const size_t idx = modelFile.rfind('.');
    std::string outFile = modelFile.substr(0, idx);

    spdlog::info("File saved to {}", outFile + ".engine");

    std::fstream os(outFile + ".engine", std::ios::out | std::ios::binary);
    os.write((const char*)serMem->data(), serMem->size());
    os.close();

    serMem->destroy();
    parser->destroy();
    engine->destroy();
    builder->destroy();
    if(CUDA_FAILED(cudaDeviceReset())){
        spdlog::error("Cant reset the device !");
    }
    spdlog::drop_all();
    return true;
}
