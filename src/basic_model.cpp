//
// Created by alex on 04.09.19.
//

#include <common.h>
#include "basic_model.h"
#include "cudaMappedMemory.h"

#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <iostream>
#include <map>
#include <memory>

using namespace nvinfer1;
using namespace std;

class Logger : public nvinfer1::ILogger
{
    void log( Severity severity, const char* msg ) override
    {
        if( severity < Severity::kWARNING /*|| mEnableDebug*/ )
            spdlog::info("[TRT]   " "{}", msg);
    }
} gLogger;

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

std::vector<std::map<std::string, uint32_t >> BasicModel::getOutputDims() const{
    std::vector<std::map<std::string, uint32_t >> out;
    out.reserve(output_tensors.size());
    for(const auto& x: output_tensors){
        out.push_back(
                {
                        {"width", DIMS_W(x.dims)},
                        {"height", DIMS_H(x.dims)},
                        {"channels", DIMS_C(x.dims)}
                });
    }
    return out;
}

// LoadNetwork
bool BasicModel::LoadNetwork(
        const std::string& model_path
        )
{
    spdlog::info("TensorRT version {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    static bool loadedPlugins = false;
    if( !loadedPlugins )
    {
        spdlog::info(LOG_TRT "loading NVIDIA plugins...");

        loadedPlugins = initLibNvInferPlugins(&gLogger, "");

        if( !loadedPlugins ) {
            spdlog::error(LOG_TRT "failed to load NVIDIA plugins");
            return false;
        }
        else
            spdlog::info(LOG_TRT "completed loading NVIDIA plugins");
    }

    cache_engine_path = model_path;
    spdlog::info(LOG_TRT "attempting to open engine cache file {}", cache_engine_path.c_str());

    std::ifstream cache(model_path, std::ios::binary);

    if( !cache )
    {
        spdlog::error(LOG_TRT "cache file not found, profiling network model on device");
        return false;
    }

    cache.seekg(0, cache.end);
    const int length = cache.tellg();
    cache.seekg(0, cache.beg);

    auto modelMem = make_unique<char[]>(length);

    if( !modelMem )
    {
        spdlog::error(LOG_TRT "failed to allocate {} bytes to deserialize model", length);
        return false;
    }

    cache.read(modelMem.get(), length);
    cache.close();
    spdlog::info(LOG_TRT "device {} loaded", model_path.c_str());

    infer = makeObjGuard<IRuntime>(createInferRuntime(gLogger));

    if( !infer )
    {
        spdlog::error(LOG_TRT "failed to create InferRuntime");
        return false;
    }

    ICudaEngine* temp_engine = infer->deserializeCudaEngine(modelMem.get(), length, nullptr);
    if(temp_engine == nullptr){
        spdlog::error(LOG_TRT "unsupported model file was provided!");
        return false;
    }
    engine = makeObjGuard<ICudaEngine>(temp_engine);
    std::size_t maxBatchSize = engine->getMaxBatchSize();

    if( !engine )
    {
        spdlog::error(LOG_TRT "failed to create CUDA engine");
        return false;
    }

    context = makeObjGuard<IExecutionContext>(engine->createExecutionContext());
    if( !context )
    {
        spdlog::error(LOG_TRT "failed to create execution context");
        return false;
    }

    spdlog::info(LOG_TRT "CUDA engine context initialized with {} bindings", engine->getNbBindings());

//    SetStream(stream);
    const int numBindings = engine->getNbBindings();

    for( int n=0; n < numBindings; n++ )
    {
        spdlog::info(LOG_TRT "binding -- index   {}", n);

        const char* bind_name = engine->getBindingName(n);

        spdlog::info("               -- name    '{}'", bind_name);
        spdlog::info("               -- in/out  {}", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

        const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

        spdlog::info("               -- # dims  {}", bind_dims.nbDims);

        for( int i=0; i < bind_dims.nbDims; i++ )
            spdlog::info("               -- dim #{}  {} \n", i, bind_dims.d[i]);
    }

    for( int n=0; n < numBindings; n++ )
    {
        string bindingName = engine->getBindingName(n);
        if(engine->bindingIsInput(n)){
            spdlog::info(LOG_TRT "binding to input 0 {}  binding index:  {}", bindingName, n);

            nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(n));

            size_t inputSize = maxBatchSize * volume(inputDims) * sizeof(float);
            spdlog::info(LOG_TRT "binding to input 0 {}  dims (b={} c={} h={} w={}) size={}",
                         bindingName,
                         maxBatchSize,
                         DIMS_C(inputDims),
                         DIMS_H(inputDims),
                         DIMS_W(inputDims),
                         inputSize);

            if(input_tensor != nullptr){
                spdlog::error("Engine must have only one input binding!");
                exit(-1);
            }
            input_tensor = make_unique<MemoryMapped>(inputSize);

            input_width        = DIMS_W(inputDims);
            input_height       = DIMS_H(inputDims);
            mInputDims = inputDims;
        } else{
            spdlog::info(LOG_TRT "binding to output {}  binding index:  {}", bindingName, n);

            nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(n));

            size_t outputSize = maxBatchSize * volume(outputDims) * sizeof(float);
            spdlog::info(LOG_TRT "binding to output {} {}  dims (b={} c={} h={} w={}) size={}",
                         n, bindingName, maxBatchSize, DIMS_C(outputDims),
                         DIMS_H(outputDims), DIMS_W(outputDims), outputSize);

            output_tensors.push_back({
                make_unique<MemoryMapped>(outputSize),
                outputDims
            });
        }
    }

//    model_path = model_path;

    spdlog::info("{} initialized.\n", model_path.c_str());
    return true;
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
        case NUM_PRECISIONS: return "INVALID PRECISION: NUM PRECISION";
    }
}

void fill_profile(IOptimizationProfile* profile, ITensor* layer, int maxBatchSize){
    Dims dims = layer->getDimensions();
    int c,h,w;
    c = DIMS_C(dims);
    h = DIMS_H(dims);
    w = DIMS_W(dims);

    profile->setDimensions(layer->getName(), OptProfileSelector::kMIN, Dims4{1, c, h, w});
    profile->setDimensions(layer->getName(), OptProfileSelector::kOPT,
            Dims4{(int)(maxBatchSize / 2 + 1), c, h, w});
    profile->setDimensions(layer->getName(), OptProfileSelector::kMAX, Dims4{maxBatchSize, c, h, w});
}

bool check_onnx_support(const string& modelFile){
    auto builder = makeObjGuard<IBuilder>(createInferBuilder(gLogger));
    if(!builder){
        throw runtime_error("failed to create builder");
    }
    builder->setMaxBatchSize(1);

    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeObjGuard<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if(!network){
        throw runtime_error("failed to create Network");
    }

    auto parser = makeObjGuard<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    if( !parser )
        throw runtime_error("failed to create ONNX Parser instance");


    std::ifstream cache(modelFile, std::ios::in | std::ios::binary);
    if( !cache )
        throw runtime_error("Cant open ONNX file");

    cache.seekg(0, cache.end);
    const int length = cache.tellg();
    cache.seekg(0, cache.beg);

    auto modelMem = make_unique<char[]>(length);
//    char* modelMem = (char*)malloc(length);

    cache.read(modelMem.get(), length);
    cache.close();

    SubGraphCollection_t sub_graph;
    return parser->supportsModel((void const *)modelMem.get(), length, sub_graph);;
}

bool _convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 const std::tuple<float, float, float>& scale,
                 const std::tuple<float, float, float>& shift,
                 int max_batch_size,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 const deviceType& device,
                 precisionType precision,
                 const pixelFormat& format)
{
    // create API root class - must span the lifetime of the engine usage
    spdlog::info(LOG_TRT "creating builder");
    auto builder = makeObjGuard<IBuilder>(createInferBuilder(gLogger));
    if(!builder){
        throw runtime_error("failed to create builder");
    }
    builder->setMaxBatchSize(max_batch_size);

    spdlog::info(LOG_TRT "creating network");
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeObjGuard<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if(!network){
        throw runtime_error("failed to create Network");
    }

    spdlog::info(LOG_TRT "loading {}", modelFile.c_str());

    auto parser = makeObjGuard<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    if( !parser )
        throw runtime_error("failed to create ONNX Parser instance");

    bool parsed = parser->parseFromFile(modelFile.c_str(), (int)ILogger::Severity::kERROR);
    if( !parsed)
        throw runtime_error("failed to parse ONNX model");

    spdlog::info(LOG_TRT "filling optimization profile");
    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < network->getNbInputs(); i++){
        auto layer = network->getInput(i);
        fill_profile(profile, layer, max_batch_size);
    }


    auto config = makeObjGuard<IBuilderConfig>(builder->createBuilderConfig());
    if(! config){
        throw runtime_error("failed to create config for model");
    }
    config->setMinTimingIterations(3);
    config->setAvgTimingIterations(2);
    config->setMaxWorkspaceSize(16_MiB);

    Dims InputDims = network->getInput(0)->getDimensions();
    spdlog::info(LOG_TRT "configuring CUDA engine");
    auto calibrator = std::make_unique<EntropyCalibrator>(
            max_batch_size,
            file_list, InputDims,
            make_float3(std::get<0>(scale), std::get<1>(scale), std::get<2>(scale)),
            make_float3(std::get<0>(shift), std::get<1>(shift), std::get<2>(shift)));


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
    config->addOptimizationProfile(profile);

    if (precision == TYPE_FP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (precision == TYPE_INT8)
    {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

    spdlog::info(LOG_TRT "trying to enable DLA...");
    int useDLACore = builder->getNbDLACores();
    if(useDLACore != 0){
        enableDLA(builder.get(), config.get(), useDLACore);
        spdlog::info(LOG_TRT "DLA enabled with {} cores", useDLACore);
        if(allowGPUFallback)
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
    } else spdlog::info(LOG_TRT "No DLA cores found(");
    // Build the engine.
    spdlog::info(LOG_TRT "building CUDA engine...");
    auto engine = makeObjGuard<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));


    if( !engine )
    {
        throw runtime_error("failed to build CUDA engine");
    }

    spdlog::info(LOG_TRT "completed building CUDA engine");
    auto serMem = makeObjGuard<IHostMemory>(engine->serialize());

    if( !serMem )
    {
        throw runtime_error("failed to serialize CUDA engine");
    }


    const size_t idx = modelFile.rfind('.');
    std::string outFile = modelFile.substr(0, idx);

    spdlog::info("File saved to {}", outFile + ".engine");

    std::fstream os(outFile + ".engine", std::ios::out | std::ios::binary);
    os.write((const char*)serMem->data(), serMem->size());
    os.close();
    return true;
}

bool convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 const std::tuple<float, float, float>& scale,
                 const std::tuple<float, float, float>& shift,
                 int max_batch_size,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 const deviceType& device,
                 precisionType precision,
                 const pixelFormat& format,
                 const std::string& logs_path){
    spdlog::flush_on(spdlog::level::info);
    if(!logs_path.empty()) {
        auto logger = spdlog::get("convertion_logger");
        if (logger == nullptr)
            logger = spdlog::rotating_logger_mt("convertion_logger", logs_path, 1048576 * 5, 1);
        spdlog::set_default_logger(logger);
        spdlog::set_error_handler([](const std::string &msg) {});
    }

    bool ret;
    try{
        bool supported = check_onnx_support(modelFile);
        if( !supported )
            throw runtime_error("Such ONNX is not supported by TRT");

        ret = _convertONNX(modelFile, file_list, scale, shift, max_batch_size,
                     allowGPUFallback, device, precision, format);
    } catch (exception& e) {
        spdlog::error(e.what());
        ret = false;
    }

    if(CUDA_FAILED(cudaDeviceReset())){
        spdlog::error("failed to reset the device");
    }
    return ret;

}
