//
// Created by alex on 04.09.19.
//

#ifndef OPENPOSETENSORRT_OPENPOSE_CUH_H
#define OPENPOSETENSORRT_OPENPOSE_CUH_H
#include "cudaUtility.h"


/*
 * Downsample to RGB or BGR, NCHW format
 */
cudaError_t cudaPreImageNetRGB( float3* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );
cudaError_t cudaPreImageNetBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );


cudaError_t cudaPreImageNetScaleShiftRGB(
        float3* input,
        size_t inputWidth,
        size_t inputHeight,
        float* output,
        size_t outputWidth,
        size_t outputHeight,
        size_t batchSize,
        float3 scale,
        float3 shift,
        cudaStream_t stream
        );

/*
 * Downsample and apply mean pixel subtraction, NCHW format
 */
cudaError_t cudaPreImageNetMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );
cudaError_t cudaPreImageNetMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, NCHW format
 */
cudaError_t cudaPreImageNetNormRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );
cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, mean pixel subtraction and standard deviation, NCHW format
 */
cudaError_t cudaPreImageNetNormMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream );
cudaError_t cudaPreImageNetNormMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream );

#endif //OPENPOSETENSORRT_OPENPOSE_CUH_H
