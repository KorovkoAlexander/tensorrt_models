#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cudaMappedMemory.h>

using namespace nvinfer1;
using namespace plugin;

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif


#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            spdlog::error("Cuda failure: {}", ret);                                                                    \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            spdlog::error(errMsg);                                                                                     \
            spdlog::error("Error in {}, function {}(), line {}", __FILE__ , FN_NAME , __LINE__ );                      \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

#define OBJ_GUARD(A) std::unique_ptr<A, void (*)(A * t)>

template <typename T, typename T_>
OBJ_GUARD(T)
makeObjGuard(T_* t)
{
    CHECK(!(std::is_base_of<T, T_>::value || std::is_same<T, T_>::value));
    auto deleter = [](T* t) {
        t->destroy();
    };
    return std::unique_ptr<T, decltype(deleter)>{static_cast<T*>(t), deleter};
}

constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

// Swaps endianness of an integral type.
class HostMemory : public IHostMemory
{
public:
    HostMemory() = delete;
    void* data() const noexcept override
    {
        return mData;
    }
    std::size_t size() const noexcept override
    {
        return mSize;
    }
    DataType type() const noexcept override
    {
        return mType;
    }

protected:
    HostMemory(std::size_t size, DataType type)
        : mSize(size)
        , mType(type)
    {
    }
    void* mData;
    std::size_t mSize;
    DataType mType;
};

template <typename ElemType, DataType dataType>
class TypedHostMemory : public HostMemory
{
public:
    explicit TypedHostMemory(std::size_t size)
        : HostMemory(size, dataType)
    {
        mData = new ElemType[size];
    };
    void destroy() noexcept override
    {
        delete[](ElemType*) mData;
        delete this;
    }
    ElemType* raw() noexcept
    {
        return static_cast<ElemType*>(data());
    }
};

using FloatMemory = TypedHostMemory<float, DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, DataType::kINT8>;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        spdlog::error(LOG_CUDA "Out of memory");
        exit(1);
    }
    return deviceMem;
}

inline bool safeCudaDealloc(void* ptr)
{
    if(ptr != nullptr){
        CHECK(cudaFree(ptr));
    }
    return true;
}


struct CudaDeleter{
    template <typename T>
    void operator() (T* obj) const {
        safeCudaDealloc(obj);
    }
};

template <typename T>
inline std::unique_ptr<T, CudaDeleter> MakeUniqueCuda(size_t memSize){
    return std::unique_ptr<T, CudaDeleter>(safeCudaMalloc(memSize));
}

template <typename T>
inline std::shared_ptr<T> MakeSharedCuda(size_t memSize){
    return std::shared_ptr<T>(safeCudaMalloc(memSize), [=](T* obj){cudaFree(obj);});
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> MakeShared(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

class MemoryMapped
{
public:
    explicit MemoryMapped(std::size_t size): size(size)
    {
        if(CUDA_FAILED(cudaHostAlloc(&pHost, size, cudaHostAllocMapped))){
            throw std::runtime_error("failed to allocate mapped memory");
        }

        if(CUDA_FAILED(cudaHostGetDevicePointer(&pDevice, pHost, 0))){
            throw std::runtime_error("Cant get device memory");
        }

    };
    ~MemoryMapped()
    {
        if(CUDA_FAILED(cudaFreeHost(pHost))){
            spdlog::error("failed to deallocate mapped memory");
            exit(-1);
        }
    }

    void* host() const{
        return pHost;
    }

    void* device() const{
        return pDevice;
    }

    size_t get_size() const{
        return size;
    }

private:
    void* pHost= nullptr;
    void* pDevice= nullptr;
    const size_t size;
};


class GPUBuffer
{
public:
    explicit GPUBuffer(std::size_t size): size(size)
    {
        data = safeCudaMalloc(size);
    };
    ~GPUBuffer()
    {
        bool res = safeCudaDealloc(data);
        if(!res){
            spdlog::error("[CUDA] " "cant dealloc cuda mapped memory");
            exit(-1);
        }
    }

    void* ptr() const{
        return data;
    }

    size_t get_size() const{
        return size;
    }

private:
    void* data;
    size_t size;
};


class CPUBuffer
{
public:
    explicit CPUBuffer(std::size_t size): size(size)
    {
        data = malloc(size);
    };

    ~CPUBuffer()
    {
        free(data);
    }

    void* ptr() const{
        return data;
    }

    size_t get_size() const{
        return size;
    }

private:
    void* data;
    size_t size;
};
// Ensures that every tensor used by a network has a scale.
//
// All tensors in a network must have a range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales for the entire network.
//
// If a tensor does not have a scale, it is assigned inScales or outScales as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its scale is assigned inScales.
// * Otherwise its scale is assigned outScales.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where scaling factors are asymmetric.
inline void setAllTensorScales(INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            spdlog::error("Trying to use DLA core {} on a platform that doesn't have any DLA cores", useDLACore);
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if ( !config->getFlag(BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    }
}

inline cudaStream_t cudaCreateStream( bool nonBlocking )
{
    uint32_t flags = cudaStreamDefault;

    if( nonBlocking )
        flags = cudaStreamNonBlocking;

    cudaStream_t stream = nullptr;

    if( CUDA_FAILED(cudaStreamCreateWithFlags(&stream, flags)) )
        return nullptr;
    return stream;
}

inline cudaError_t setDevice(int device) {
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
    return error;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int DIMS_C(const Dims& dims){
    int C = dims.nbDims >= 3 ? dims.d[dims.nbDims - 3] : 1;
    return C > 0? C : 1;
}
inline int DIMS_H(const Dims& dims){
    int H = dims.nbDims >= 2 ? dims.d[dims.nbDims - 2] : 1;
    return H > 0? H : 1;
}
inline int DIMS_W(const Dims& dims){
    int W = dims.nbDims >= 1 ? dims.d[dims.nbDims - 1] : 1;
    return W > 0? W : 1;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    int64_t out = 1;
    for(int i = 0;  i < d.nbDims - 1; i++){
        int Size = d.d[d.nbDims - 1 - i];
        out *= Size > 0 ? Size : 1;
    }
    return out;
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kBOOL:
    case DataType::kINT8: return 1;
    }
    return 0;
}



inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}

#endif // TENSORRT_COMMON_H