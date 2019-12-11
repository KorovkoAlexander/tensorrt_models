//
// Created by alex on 04.09.19.
// from https://github.com/dusty-nv/jetson-inference/blob/master/c/imageNet.cu
//
#include <Utils.h>

__global__ void gpuPreImageNetRGB(float2 scale, float3* input, int iWidth, float* output, int oWidth, int oHeight )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float3 px  = input[ dy * iWidth + dx ];
//    const float3 rgb = make_float3(px.x, px.y, px.z);

    output[n * 0 + m] = px.x;
    output[n * 1 + m] = px.y;
    output[n * 2 + m] = px.z;
}


// cudaPreImageNetRGB
cudaError_t cudaPreImageNetRGB( float3* input, size_t inputWidth, size_t inputHeight,
                                float* output, size_t outputWidth, size_t outputHeight,
                                cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

//    std::cerr << iDivUp(outputWidth,blockDim.x) << " " << iDivUp(outputHeight,blockDim.y) << std::endl;

    gpuPreImageNetRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

    return CUDA(cudaGetLastError());
}

__global__ void gpuPreImageNetScaleShiftRGB(
        float2 scale, float3* input,
        int iWidth, int iHeight, float* output,
        int oWidth, int oHeight, int batchSize,
        float3 _scale, float3 _shift
        ){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int k = 3 * oWidth * oHeight;
    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    for(int b = 0; b < batchSize; b++){
        const float3 px  = input[b*iWidth*iHeight +  dy * iWidth + dx ];

        output[k*b + n * 0 + m] = px.x/_scale.x - _shift.x;
        output[k*b + n * 1 + m] = px.y/_scale.y - _shift.y;
        output[k*b + n * 2 + m] = px.z/_scale.z - _shift.z;
    }
}


// cudaPreImageNetRGB
cudaError_t cudaPreImageNetScaleShiftRGB( float3* input, size_t inputWidth, size_t inputHeight,
                                float* output, size_t outputWidth, size_t outputHeight, size_t batchSize,
                                float3 _scale, float3 _shift,
                                cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

//    std::cerr << iDivUp(outputWidth,blockDim.x) << " " << iDivUp(outputHeight,blockDim.y) << std::endl;

    gpuPreImageNetScaleShiftRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, batchSize, _scale, _shift);

    return CUDA(cudaGetLastError());
}

// gpuPreImageNetBGR
__global__ void gpuPreImageNetBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.z, px.y, px.x);

    output[n * 0 + m] = bgr.x;
    output[n * 1 + m] = bgr.y;
    output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetBGR
cudaError_t cudaPreImageNetBGR( float4* input, size_t inputWidth, size_t inputHeight,
                                float* output, size_t outputWidth, size_t outputHeight,
                                cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetMeanRGB
__global__ void gpuPreImageNetMeanRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.x - mean_value.x, px.y - mean_value.y, px.z - mean_value.z);

    output[n * 0 + m] = bgr.x;
    output[n * 1 + m] = bgr.y;
    output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetMeanRGB
cudaError_t cudaPreImageNetMeanRGB( float4* input, size_t inputWidth, size_t inputHeight,
                                    float* output, size_t outputWidth, size_t outputHeight,
                                    const float3& mean_value, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetMeanRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetMeanBGR
__global__ void gpuPreImageNetMeanBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.z - mean_value.x, px.y - mean_value.y, px.x - mean_value.z);

    output[n * 0 + m] = bgr.x;
    output[n * 1 + m] = bgr.y;
    output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetMeanBGR
cudaError_t cudaPreImageNetMeanBGR( float4* input, size_t inputWidth, size_t inputHeight,
                                    float* output, size_t outputWidth, size_t outputHeight,
                                    const float3& mean_value, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetMeanBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormRGB
__global__ void gpuPreImageNetNormRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.x, px.y, px.z);

    output[n * 0 + m] = bgr.x * multiplier + min_value;
    output[n * 1 + m] = bgr.y * multiplier + min_value;
    output[n * 2 + m] = bgr.z * multiplier + min_value;
}


// cudaPreImageNetNormRGB
cudaError_t cudaPreImageNetNormRGB( float4* input, size_t inputWidth, size_t inputHeight,
                                    float* output, size_t outputWidth, size_t outputHeight,
                                    const float2& range, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    const float multiplier = (range.y - range.x) / 255.0f;

    //printf("cudaPreImageNetNorm([%f, %f])  multiplier=%f\n", range.x, range.y, multiplier);

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetNormRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormBGR
__global__ void gpuPreImageNetNormBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.z, px.y, px.x);

    output[n * 0 + m] = bgr.x * multiplier + min_value;
    output[n * 1 + m] = bgr.y * multiplier + min_value;
    output[n * 2 + m] = bgr.z * multiplier + min_value;
}


// cudaPreImageNetNorm
cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight,
                                    float* output, size_t outputWidth, size_t outputHeight,
                                    const float2& range, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    const float multiplier = (range.y - range.x) / 255.0f;

    //printf("cudaPreImageNetNorm([%f, %f])  multiplier=%f\n", range.x, range.y, multiplier);

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetNormBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);

    return CUDA(cudaGetLastError());
}



// gpuPreImageNetNormMeanRGB
__global__ void gpuPreImageNetNormMeanRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.x * multiplier + min_value, px.y * multiplier + min_value, px.z * multiplier + min_value);

    output[n * 0 + m] = (bgr.x - mean.x) / stdDev.x;
    output[n * 1 + m] = (bgr.y - mean.y) / stdDev.y;
    output[n * 2 + m] = (bgr.z - mean.z) / stdDev.z;
}


// cudaPreImageNetNormMeanRGB
cudaError_t cudaPreImageNetNormMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    const float multiplier = (range.y - range.x) / 255.0f;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetNormMeanRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x, mean, stdDev);

    return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormMeanBGR
__global__ void gpuPreImageNetNormMeanBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= oWidth || y >= oHeight )
        return;

    const int n = oWidth * oHeight;
    const int m = y * oWidth + x;

    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    const float4 px  = input[ dy * iWidth + dx ];
    const float3 bgr = make_float3(px.z * multiplier + min_value, px.y * multiplier + min_value, px.x * multiplier + min_value);

    output[n * 0 + m] = (bgr.x - mean.x) / stdDev.x;
    output[n * 1 + m] = (bgr.y - mean.y) / stdDev.y;
    output[n * 2 + m] = (bgr.z - mean.z) / stdDev.z;
}


// cudaPreImageNetNormMeanBGR
cudaError_t cudaPreImageNetNormMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    const float multiplier = (range.y - range.x) / 255.0f;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreImageNetNormMeanBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x, mean, stdDev);

    return CUDA(cudaGetLastError());
}

