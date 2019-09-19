//
// Created by alex on 12.09.19.
//

#include "EntropyCalibrator.h"
#include <iterator>
#include <assert.h>

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

    int num_images = 0;
    while(infile >> tmp) {
        _fnames.push_back(tmp);
        num_images++;
    }

    std:: cout << "Found " << num_images <<" images!" << std::endl;

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

    assert(channels == 3);
    unsigned char* img  = loadImageIO(fname.c_str(), &width, &height, &channels);

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
            _batch[offset+2] = float(img[offset])/scale.x - shift.x;
            _batch[offset+1] = float(img[offset+1])/scale.y - shift.y;
            _batch[offset] = float(img[offset+2])/scale.z - shift.z;

        }
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
