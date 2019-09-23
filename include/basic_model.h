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
#include <math.h>
#include <tuple>

#include <EntropyCalibrator.h>


typedef nvinfer1::DimsCHW Dims3;
typedef nvinfer1::DimsNCHW Dims4;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]


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
    std::string name;
    Dims3 dims;
    uint32_t size;
    float* CPU;
    float* CUDA;
};

bool loadImage(uint8_t * img, float3** cpu, const int& imgWidth, const int& imgHeight, const int& batchSize);

bool DetectNativePrecision( const std::vector<precisionType>& types, precisionType type );

const char* precisionTypeToStr( precisionType type );

std::vector<precisionType> DetectNativePrecisions( deviceType device );

precisionType FindFastestPrecision( deviceType device, bool allowInt8 );

bool convertONNX(const std::string& modelFile, // name for model
                 const std::string& file_list,
                 const std::tuple<float, float, float>& scale,
                 const std::tuple<float, float, float>& shift,
                 unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
                 bool allowGPUFallback,
                 const deviceType& device = DEVICE_GPU,
                 precisionType precision = TYPE_FP32,
                 const pixelFormat& format = BGR);


class BasicModel
{
public:
    virtual ~BasicModel();

    bool LoadNetwork( const std::string& model,
                      const std::string& input_blob="data", const std::string& output_blob="prob",
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
                      deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
                      cudaStream_t stream=nullptr);

    bool LoadNetwork( const std::string& model,
                      const std::string& input_blob,
                      const std::vector<std::string>& output_blobs,
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
                      deviceType device=DEVICE_GPU,
                      bool allowGPUFallback=true,
                      cudaStream_t stream=nullptr);

    bool LoadNetwork( const std::string& model,
                      const std::string& input_blob,
                      const Dims3& input_dims,
                      const std::vector<std::string>& output_blobs,
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
                      deviceType device=DEVICE_GPU,
                      bool allowGPUFallback=true,
                      cudaStream_t stream=nullptr);

    inline bool AllowGPUFallback() const { return mAllowGPUFallback; }

    inline deviceType GetDevice() const	{ return mDevice; }

    inline outputLayer get_output(const int& n) const { return mOutputs[n];}

    inline cudaStream_t GetStream() const { return mStream; }

    cudaStream_t CreateStream( bool nonBlocking=true );

    void SetStream( cudaStream_t stream );

    inline const char* GetModelPath() const	{ return mModelPath.c_str(); }

    inline std::map<std::string, uint32_t > getInputDims() const {return {{"width", mWidth}, {"height", mHeight}};};
    std::vector<std::map<std::string, uint32_t >> getOutputDims() const;

protected:

    BasicModel();

protected:

    /* Member Variables */
    std::string mModelPath;
    std::string mInputBlobName;
    std::string mCacheEnginePath;

    deviceType    mDevice;
    cudaStream_t  mStream;

    nvinfer1::IRuntime* mInfer;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;

    uint32_t mWidth;
    uint32_t mHeight;
    uint32_t mInputSize;
    float*   mInputCPU;
    float*   mInputCUDA;
    uint32_t mMaxBatchSize;
    bool	    mAllowGPUFallback;

    Dims3 mInputDims;

    std::vector<outputLayer> mOutputs;
};

#endif //BASIC_MODEL_H
