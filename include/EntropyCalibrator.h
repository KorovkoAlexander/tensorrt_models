//
// Created by alex on 12.09.19.
//

#ifndef TENSORRT_MODELS_ENTROPYCALIBRATOR_H
#define TENSORRT_MODELS_ENTROPYCALIBRATOR_H

#include <NvInfer.h>
#include <common.h>
#include <imageIO.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <tuple>
#include <memory>

enum pixelFormat
{
    BGR = 0,
    RGB
};

class EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    EntropyCalibrator(
            int max_batch_size,
            const std::string& file_list,
            const Dims& inputDims,
            const float3& scale = {58.395, 57.12 , 57.375},
            const float3& shift = {123.675, 116.28 , 103.53 },
            const pixelFormat& format = RGB);

    ~EntropyCalibrator() override ;

    int getBatchSize() const override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    const void* readCalibrationCache(size_t& length) override;

    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    static std::string calibrationTableName();
    size_t mInputCount1;
    Dims input_dims;
    void* mDeviceInput1 { nullptr };
    int max_batch_size;

    float3 scale;
    float3 shift;

    int _cur_id;
    std::string _file_list;
    std::vector<std::string> _fnames;
    float* _batch = nullptr;
//    nvinfer1::DimsCHW dims;
    pixelFormat format;
};


#endif //TENSORRT_MODELS_ENTROPYCALIBRATOR_H
