//
// Created by alex on 04.09.19.
// based on https://github.com/dusty-nv/jetson-inference/blob/e12e6e64365fed83e255800382e593bf7e1b1b1a/tensorNet.cpp
//

#ifndef BASIC_MODEL_H
#define BASIC_MODEL_H

#include <NvInfer.h>

#include <cudaUtility.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <tuple>

#include <EntropyCalibrator.h>

using Dims3 = nvinfer1::Dims3;
using Dims4 = nvinfer1::Dims4;


#define DEFAULT_MAX_BATCH_SIZE  1
#define LOG_TRT "[TRT]   "

enum deviceType
{
    DEVICE_GPU = 0,			/**< GPU (if multiple GPUs are present, a specific GPU can be selected with cudaSetDevice() */
    DEVICE_DLA,				/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
    DEVICE_DLA_0 = DEVICE_DLA,	/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
    DEVICE_DLA_1,				/**< Deep Learning Accelerator (DLA) Core 1 (only on Jetson Xavier) */
    NUM_DEVICES				/**< Number of device types defined */
};

enum precisionType
{
    TYPE_DISABLED = 0,	/**< Unknown, unspecified, or disabled type */
    TYPE_FASTEST,		/**< The fastest detected precision should be use (i.e. try INT8, then FP16, then FP32) */
    TYPE_FP32,		/**< 32-bit floating-point precision (FP32) */
    TYPE_FP16,		/**< 16-bit floating-point half precision (FP16) */
    TYPE_INT8,		/**< 8-bit integer precision (INT8) */
    NUM_PRECISIONS		/**< Number of precision types defined */
};

struct outputLayer
{
    outputLayer(std::unique_ptr<GPUBuffer> memory, Dims dims): memory(std::move(memory)), dims(dims){}
    Dims dims;
    std::unique_ptr<GPUBuffer> memory{nullptr};
};

const char* precisionTypeToStr( precisionType type );

bool convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 const std::tuple<float, float, float>& scale,
                 const std::tuple<float, float, float>& shift,
                 int max_batch_size,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 const deviceType& device = DEVICE_GPU,
                 precisionType precision = TYPE_FP32,
                 const pixelFormat& format = BGR,
                 const std::string& logs_path= {});

void fill_profile(IOptimizationProfile* profile, ITensor* layer, int maxBatchSize);


class BasicModel
{
public:
    virtual ~BasicModel() = default;

    bool LoadNetwork(const std::string& model);

    inline std::map<std::string, uint32_t > getInputDims() const {
        return {{"width", input_width}, {"height", input_height}};
    };

    std::vector<std::map<std::string, uint32_t >> getOutputDims() const;

protected:

    BasicModel() = default;

protected:

    /* Member Variables */
    std::string cache_engine_path;

    std::shared_ptr<nvinfer1::IRuntime> infer {nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> engine {nullptr};
    std::shared_ptr<nvinfer1::IExecutionContext> context {nullptr};

    int input_width=0;
    int input_height=0;
    std::unique_ptr<GPUBuffer> input_tensor {nullptr};
    Dims mInputDims;

    std::vector<outputLayer> output_tensors;
};

#endif //BASIC_MODEL_H
