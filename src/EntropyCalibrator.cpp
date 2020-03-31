//
// Created by alex on 12.09.19.
//

#include "EntropyCalibrator.h"
#include "spdlog/spdlog.h"
#include <iterator>
#include <cassert>

using namespace std;

EntropyCalibrator::EntropyCalibrator(
        int max_batch_size,
        const std::string& file_list,
        const Dims& inputDims,
        const float3& scale,
        const float3& shift,
        const pixelFormat& format):
        max_batch_size(max_batch_size), _file_list(file_list),
        _cur_id(0), scale(scale), shift(shift), format(format), input_dims(inputDims)
{
    if (file_list.empty()){
        return;
    }
    _batch = new float[max_batch_size*DIMS_W(inputDims) * DIMS_H(inputDims) * DIMS_C(inputDims)];

    std::ifstream infile(file_list);
    std::string tmp;

    int num_images = 0;
    while(infile >> tmp) {
        _fnames.push_back(tmp);
        num_images++;
    }

    spdlog::info("[CALIB] ""Found {} images!", num_images);

//    dims = nvinfer1::DimsCHW(DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims));
    mInputCount1 = max_batch_size*DIMS_C(inputDims)* DIMS_H(inputDims)* DIMS_W(inputDims);
    cudaMalloc(&mDeviceInput1, mInputCount1 * sizeof(float));
};

EntropyCalibrator::~EntropyCalibrator()
{
    delete[] _batch;
    cudaFree(mDeviceInput1);
};

int EntropyCalibrator::getBatchSize() const { return 1;};

bool EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    for(int b = 0; b < max_batch_size; b++){
        if(_cur_id + 1 >= _fnames.size()){
            return false;
        }

        _cur_id+=1;
        std::string fname = _fnames[_cur_id];
        int width = DIMS_W(input_dims);
        int height = DIMS_H(input_dims);
        int channels = DIMS_C(input_dims);

        unsigned char* img = loadImageIO(fname.c_str(), &width, &height, &channels);
        assert(channels == 3);

        while (img == nullptr){
            if(_cur_id + 1 >= _fnames.size()){
                return false;
            }

            _cur_id+=1;
            fname = _fnames[_cur_id];
            img  = loadImageIO(fname.c_str(), &width, &height, &channels);
        }

        for( int y=0; y < height; y++ )
        {
            const size_t yOffset = y * width * channels * sizeof(uint8_t);
            for( int x=0; x < width; x++ )
            {

                const size_t offset = yOffset + x * channels * sizeof(uint8_t);
                const size_t batch_offset = b * width * height * channels;
                const size_t channel_offset = width * height;
                switch (format){
                    case BGR:
                        _batch[batch_offset+2*channel_offset + y*width + x] = (float(img[offset]) - shift.x)/scale.x;
                        _batch[batch_offset+1*channel_offset + y*width + x] = (float(img[offset+1]) - shift.y)/scale.y;
                        _batch[batch_offset+0*channel_offset + y*width + x] = (float(img[offset+2]) - shift.z)/scale.z;
                        break;
                    case RGB:
                        _batch[batch_offset+0*channel_offset + y*width + x] = (float(img[offset]) - shift.x)/scale.x;
                        _batch[batch_offset+1*channel_offset + y*width + x] = (float(img[offset+1]) - shift.y)/scale.y;
                        _batch[batch_offset+2*channel_offset + y*width + x] = (float(img[offset+2]) - shift.z)/scale.z;
                        break;
                    default:
                        spdlog::error("Bad pixelFormat!");
                        exit(-1);
                }
            }
        }
        free(img);
    }

//    _cur_id += dims.n();
    cudaMemcpy(mDeviceInput1, _batch,  mInputCount1 * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = mDeviceInput1;
    return true;
};

const void* EntropyCalibrator::readCalibrationCache(size_t& length){ return nullptr;};

void EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length){};

std::string EntropyCalibrator::calibrationTableName(){ return "CalibrationTable"; }
