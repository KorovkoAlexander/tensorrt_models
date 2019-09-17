//
// Created by alex on 04.09.19.
//

#include "basic_model.h"
#include "cudaMappedMemory.h"

#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"

#include <EntropyCalibrator.h>

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
            printf(LOG_TRT "%s\n", msg);
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

bool loadImage(uint8_t * img, float3** cpu, const int& imgWidth, const int& imgHeight){
    if( !img )
        return false;

    const int imgChannels = 3;

//    float4 mean  = make_float4(0,0,0,0);

    if (cpu == nullptr)
    {
        std::cerr << "Cant load the image! No cpu memory provided.." << std::endl;
        return false;
    }

    // convert uint8 image to float4
    float3* cpuPtr = *cpu;

    for( int y=0; y < imgHeight; y++ )
    {
        const size_t yOffset = y * imgWidth * imgChannels * sizeof(uint8_t);

        for( int x=0; x < imgWidth; x++ )
        {
            #define GET_PIXEL(channel)	    float(img[offset + channel])
            #define SET_PIXEL_FLOAT3(r,g,b) cpuPtr[y*imgWidth+x] = make_float3(r,g,b)

            const size_t offset = yOffset + x * imgChannels * sizeof(uint8_t);
            SET_PIXEL_FLOAT3(GET_PIXEL(0), GET_PIXEL(1), GET_PIXEL(2));

        }
    }
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
            std::cerr << "Cant deallocate mInputCPU in dtor!" << std::endl;
        }
        for(auto& x : mOutputs){
            if(!cudaDeallocMapped((void**)x.CPU)){
                std::cerr << "Cant deallocate mOutputs in dtor!" << std::endl;
            }
        }
        if(CUDA_FAILED(cudaDeviceReset())){
            std::cerr << "Cant reset the device in dtor!" << std::endl;
        }

}


// LoadNetwork
bool BasicModel::LoadNetwork(const char* model_path,
                             const char* input_blob,
                             const char* output_blob,
                             uint32_t maxBatchSize,
                             deviceType device,
                             bool allowGPUFallback,
                             cudaStream_t stream)
{
    std::vector<std::string> outputs;
    outputs.emplace_back(output_blob);

    return LoadNetwork(model_path, input_blob, outputs, maxBatchSize, device, allowGPUFallback );
}


// LoadNetwork
bool BasicModel::LoadNetwork(const char* model_path_,
                             const char* input_blob,
                             const std::vector<std::string>& output_blobs,
                             uint32_t maxBatchSize,
                             deviceType device, bool allowGPUFallback,
                             cudaStream_t stream)
{
    return LoadNetwork(model_path_,
                       input_blob, Dims3(1,1,1), output_blobs,
                       maxBatchSize, device,
                       allowGPUFallback, stream);
}


// LoadNetwork
bool BasicModel::LoadNetwork(
        const char* model_path_,
        const char* input_blob,
        const Dims3& input_dims,
        const std::vector<std::string>& output_blobs,
        uint32_t maxBatchSize,
        deviceType device,
        bool allowGPUFallback,
        cudaStream_t stream
        )
{
    if(!model_path_ )
        return false;

    printf(LOG_TRT "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    /*
     * load NV inference plugins
     */
    static bool loadedPlugins = false;

    if( !loadedPlugins )
    {
        printf(LOG_TRT "loading NVIDIA plugins...\n");

        loadedPlugins = initLibNvInferPlugins(&gLogger, "");

        if( !loadedPlugins )
            printf(LOG_TRT "failed to load NVIDIA plugins\n");
        else
        printf(LOG_TRT "completed loading NVIDIA plugins.\n");
    }

    const std::string model_path    = model_path_;

    /*
     * attempt to load network from cache before profiling with tensorRT
     */
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);

    mCacheEnginePath = model_path;
    printf(LOG_TRT "attempting to open engine cache file %s\n", mCacheEnginePath.c_str());

    std::ifstream cache( mCacheEnginePath );

    if( !cache )
    {
        printf(LOG_TRT "cache file not found, profiling network model on device\n");
        return false;
    }
    else
    {
        printf(LOG_TRT "loading network profile from engine cache... %s\n", mCacheEnginePath.c_str());
        gieModelStream << cache.rdbuf();
        cache.close();
    }

    printf(LOG_TRT "device %s loaded\n", model_path.c_str());


    /*
     * create runtime inference engine execution context
     */
    nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);

    if( !infer )
    {
        printf(LOG_TRT "failed to create InferRuntime\n");
        return false;
    }


    // if using DLA, set the desired core before deserialization occurs
    if( device == DEVICE_DLA_0 )
    {
        printf(LOG_TRT "enabling DLA core 0\n");
        infer->setDLACore(0);
    }
    else if( device == DEVICE_DLA_1 )
    {
        printf(LOG_TRT "enabling DLA core 1\n");
        infer->setDLACore(1);
    }


    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);

    if( !modelMem )
    {
        printf(LOG_TRT "failed to allocate %i bytes to deserialize model\n", modelSize);
        return false;
    }

    gieModelStream.read((char*)modelMem, modelSize);
    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);

    if( !engine )
    {
        printf(LOG_TRT "failed to create CUDA engine\n");
        return false;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    if( !context )
    {
        printf(LOG_TRT "failed to create execution context\n");
        return false;
    }

    printf(LOG_TRT "CUDA engine context initialized with %u bindings\n", engine->getNbBindings());

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
        printf(LOG_TRT "binding -- index   %i\n", n);

        const char* bind_name = engine->getBindingName(n);

        printf("               -- name    '%s'\n", bind_name);
        printf("               -- type    %s\n", dataTypeToStr(engine->getBindingDataType(n)));
        printf("               -- in/out  %s\n", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

        const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

        printf("               -- # dims  %i\n", bind_dims.nbDims);

        for( int i=0; i < bind_dims.nbDims; i++ )
            printf("               -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));
    }

    /*
     * determine dimensions of network input bindings
     */
    const int inputIndex = engine->getBindingIndex(input_blob);

    printf(LOG_TRT "binding to input 0 %s  binding index:  %i\n", input_blob, inputIndex);

    nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));

    size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
    printf(LOG_TRT "binding to input 0 %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n",
            input_blob,
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
        printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
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
        printf(LOG_TRT "binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

        nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));

        size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
        printf(LOG_TRT "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);

        // allocate output memory
        void* outputCPU  = nullptr;
        void* outputCUDA = nullptr;

        //if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
        if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
        {
            printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
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


    printf("%s initialized.\n", mModelPath.c_str());
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



bool DetectNativePrecision( const std::vector<precisionType>& types, precisionType type )
{
    const uint32_t numTypes = types.size();

    for( uint32_t n=0; n < numTypes; n++ )
    {
        if( types[n] == type )
            return true;
    }

    return false;
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

std::vector<precisionType> DetectNativePrecisions( deviceType device )
{
    std::vector<precisionType> types;
    Logger logger;

    // create a temporary builder for querying the supported types
    nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(logger);

    if( !builder )
    {
        printf(LOG_TRT "QueryNativePrecisions() failed to create TensorRT IBuilder instance\n");
        return types;
    }

    if( device == DEVICE_DLA_0 || device == DEVICE_DLA_1 )
        builder->setFp16Mode(true);

    builder->setDefaultDeviceType( deviceTypeToTRT(device) );

    // FP32 is supported on all platforms
    types.push_back(TYPE_FP32);

    // detect fast (native) FP16
    if( builder->platformHasFastFp16() )
        types.push_back(TYPE_FP16);

    if( builder->platformHasFastInt8() )
        types.push_back(TYPE_INT8);

    // print out supported precisions (optional)
    const uint32_t numTypes = types.size();

    for( uint32_t n=0; n < numTypes; n++ )
    {
        printf("%s", precisionTypeToStr(types[n]));

        if( n < numTypes - 1 )
            printf(", ");
    }

    printf("\n");
    builder->destroy();
    return types;
}

// FindFastestPrecision
precisionType FindFastestPrecision( deviceType device, bool allowInt8 )
{
    std::vector<precisionType> types = DetectNativePrecisions(device);

    if( allowInt8 && DetectNativePrecision(types, TYPE_INT8) )
        return TYPE_INT8;
    else if( DetectNativePrecision(types, TYPE_FP16) )
        return TYPE_FP16;
    else
        return TYPE_FP32;
}

bool convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 deviceType device,
                 precisionType precision)			   // output stream for the GIE model
{
    // create API root class - must span the lifetime of the engine usage
    nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    builder->setAverageFindIterations(2);

    printf(LOG_TRT "loading %s\n", modelFile.c_str());


    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if( !parser )
    {
        printf(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
        return false;
    }

    if( !parser->parseFromFile(modelFile.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING) )
    {
        printf(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
        return false;
    }

    // extract the dimensions of the network input blobs
    nvinfer1::Dims3 inputDimensions = static_cast<nvinfer1::Dims3&&>(network->getInput(0)->getDimensions());


    // build the engine
    printf(LOG_TRT "configuring CUDA engine\n");

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);


    // set up the builder for the desired precision
    if (precision == TYPE_FASTEST){
        if(!file_list.empty())
            precision = FindFastestPrecision(device, true);
        else
            precision = FindFastestPrecision(device, false);
    }

    if( precision == TYPE_INT8)
    {
        builder->setInt8Mode(true);

        int batch_size = 1;
        int imgWidth = DIMS_W(inputDimensions);
        int imgHeight = DIMS_H(inputDimensions);
        int imgChannels = DIMS_C(inputDimensions);

        nvinfer1::IInt8Calibrator* calibrator= new EntropyCalibrator(file_list, batch_size, imgWidth, imgHeight, imgChannels, false);

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
    printf(LOG_TRT "building FP16:  %s\n", isFp16Enabled(builder) ? "ON" : "OFF");
    printf(LOG_TRT "building INT8:  %s\n", isInt8Enabled(builder) ? "ON" : "OFF");
    printf(LOG_TRT "building CUDA engine (this may take a few minutes the first time a network is loaded)\n");

    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

    if( !engine )
    {
        printf(LOG_TRT "failed to build CUDA engine\n");
        return false;
    }

    printf(LOG_TRT "completed building CUDA engine\n");

    // we don't need the network definition any more, and we can destroy the parser
    network->destroy();
    //parser->destroy();

    nvinfer1::IHostMemory* serMem = engine->serialize();

    if( !serMem )
    {
        printf(LOG_TRT "failed to serialize CUDA engine\n");
        return false;
    }

    std::string outFile;
    std::getline(std::stringstream(modelFile), outFile, '.');
    std::cout << outFile + "_1.engine" << std::endl;

    std::ofstream gieModelStream(outFile + "_1.engine");
//    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write((const char*)serMem->data(), serMem->size());

    engine->destroy();
    builder->destroy();




    return true;
}
