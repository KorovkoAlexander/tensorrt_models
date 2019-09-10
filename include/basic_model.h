//
// Created by alex on 04.09.19.
// based on https://github.com/dusty-nv/jetson-inference/blob/e12e6e64365fed83e255800382e593bf7e1b1b1a/tensorNet.cpp
//

#ifndef OPENPOSETENSORRT_BASIC_MODEL_H
#define OPENPOSETENSORRT_BASIC_MODEL_H

#include <NvInfer.h>

#include <cudaUtility.h>
#include <vector>
#include <sstream>
#include <math.h>

typedef nvinfer1::DimsCHW Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]


#define DEFAULT_MAX_BATCH_SIZE  1
#define LOG_TRT "[TRT]   "


enum precisionType
{
    TYPE_DISABLED = 0,	/**< Unknown, unspecified, or disabled type */
    TYPE_FASTEST,		/**< The fastest detected precision should be use (i.e. try INT8, then FP16, then FP32) */
    TYPE_FP32,		/**< 32-bit floating-point precision (FP32) */
    TYPE_FP16,		/**< 16-bit floating-point half precision (FP16) */
    TYPE_INT8,		/**< 8-bit integer precision (INT8) */
    NUM_PRECISIONS		/**< Number of precision types defined */
};

const char* precisionTypeToStr( precisionType type );

precisionType precisionTypeFromStr( const char* str );

enum deviceType
{
    DEVICE_GPU = 0,			/**< GPU (if multiple GPUs are present, a specific GPU can be selected with cudaSetDevice() */
    DEVICE_DLA,				/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
    DEVICE_DLA_0 = DEVICE_DLA,	/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
    DEVICE_DLA_1,				/**< Deep Learning Accelerator (DLA) Core 1 (only on Jetson Xavier) */
    NUM_DEVICES				/**< Number of device types defined */
};

const char* deviceTypeToStr( deviceType type );

deviceType deviceTypeFromStr( const char* str );

struct outputLayer
{
    std::string name;
    Dims3 dims;
    uint32_t size;
    float* CPU;
    float* CUDA;
};

bool loadImage(uint8_t * img, float3** cpu, const int& imgWidth, const int& imgHeight);


class BasicModel
{
public:
    virtual ~BasicModel();

    bool LoadNetwork( const char* model,
                      const char* input_blob="data", const char* output_blob="prob",
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, precisionType precision=TYPE_FASTEST,
                      deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
                      cudaStream_t stream=NULL);

    bool LoadNetwork( const char* model,
                      const char* input_blob,
                      const std::vector<std::string>& output_blobs,
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
                      precisionType precision=TYPE_FASTEST,
                      deviceType device=DEVICE_GPU,
                      bool allowGPUFallback=true,
                      cudaStream_t stream=NULL);

    bool LoadNetwork( const char* model,
                      const char* input_blob,
                      const Dims3& input_dims,
                      const std::vector<std::string>& output_blobs,
                      uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
                      precisionType precision=TYPE_FASTEST,
                      deviceType device=DEVICE_GPU,
                      bool allowGPUFallback=true,
                      cudaStream_t stream=NULL);

    inline bool AllowGPUFallback() const { return mAllowGPUFallback; }

    inline deviceType GetDevice() const	{ return mDevice; }

    inline outputLayer get_output(const int& n) const { return mOutputs[n];}

    inline precisionType GetPrecision() const { return mPrecision; }

    inline bool IsPrecision( precisionType type ) const	{ return (mPrecision == type); }

    static precisionType FindFastestPrecision( deviceType device=DEVICE_GPU, bool allowInt8=true );

    static std::vector<precisionType> DetectNativePrecisions( deviceType device=DEVICE_GPU );

    static bool DetectNativePrecision( const std::vector<precisionType>& nativeTypes, precisionType type );

    static bool DetectNativePrecision( precisionType precision, deviceType device=DEVICE_GPU );

    inline cudaStream_t GetStream() const { return mStream; }

    cudaStream_t CreateStream( bool nonBlocking=true );

    void SetStream( cudaStream_t stream );

    inline const char* GetModelPath() const	{ return mModelPath.c_str(); }

protected:

    BasicModel();

    class Logger : public nvinfer1::ILogger
    {
        void log( Severity severity, const char* msg ) override
        {
            if( severity != Severity::kINFO /*|| mEnableDebug*/ )
                printf(LOG_TRT "%s\n", msg);
        }
    } gLogger;


protected:

    /* Member Variables */
    std::string mModelPath;
    std::string mInputBlobName;
    std::string mCacheEnginePath;

    deviceType    mDevice;
    precisionType mPrecision;
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

#endif //OPENPOSETENSORRT_BASIC_MODEL_H
