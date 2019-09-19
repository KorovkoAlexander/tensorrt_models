//
// Created by alex on 12.09.19.
//

#include "EntropyCalibrator.h"
#include <iterator>

EntropyCalibrator::EntropyCalibrator(
        const std::string& file_list,
        const int& width,
        const int& height,
        const int& channel,
        const float3& scale,
        const float3& shift):
        _file_list(file_list), _cur_id(0), scale(scale), shift(shift)
{
    _batch = new float[width * height * channel];

    std::ifstream infile(file_list);
    std::string tmp;

    std:: cout << "Found images!" << std::endl;
    while(infile >> tmp) {
        _fnames.push_back(tmp);
        std::cout << tmp << std::endl;
    }

    dims = nvinfer1::DimsCHW(channel, height, width);
    mInputCount1 = dims.c() * dims.h() * dims.w();
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
    if(_cur_id + 1 >= _fnames.size()){
        return false;
    }

    _cur_id+=1;
    std::string fname = _fnames[_cur_id];
    int width = dims.w();
    int height = dims.h();
    int channels = dims.c();
    unsigned char* img  = loadImageIO(fname.c_str(), &width, &height, &channels);

    for(int k = 0; k < width*height*channels; k++){
        int c = k % channels;
        float c_scale, c_shift;
        switch (c){
            case 0: //r
                c_scale = scale.x;
                c_shift = shift.x;
                break;
            case 1: // g
                c_scale = scale.y;
                c_shift = shift.y;
                break;
            case 2: // b
                c_scale = scale.z;
                c_shift = shift.z;
                break;
            default:
                std::cout << "Some problems with calibrating reading.." << std:: endl;
                exit(-1);
        }
        _batch[k] = (float)(img[k])/c_scale - c_shift;
    }

    free(img);

//    _cur_id += dims.n();
    cudaMemcpy(mDeviceInput1, _batch,  mInputCount1 * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = mDeviceInput1;
    return true;
};

const void* EntropyCalibrator::readCalibrationCache(size_t& length)
{
    return nullptr;
};

void EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{

};

std::string EntropyCalibrator::calibrationTableName()
{
    return "CalibrationTable";
}
