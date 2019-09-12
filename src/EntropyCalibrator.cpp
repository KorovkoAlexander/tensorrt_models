//
// Created by alex on 12.09.19.
//

#include "EntropyCalibrator.h"
#include <iterator>

EntropyCalibrator::EntropyCalibrator(
        const std::string& file_list,
        const int& batchSize,
        const int& width,
        const int& height,
        const int& channel,
        bool readCache):
        mReadCache(readCache),
        _file_list(file_list), _cur_id(0)
{
    _batch = new float[batchSize * width * height * channel];

    std::ifstream infile(file_list);
    std::string tmp;

    std:: cout << "Found images!" << std::endl;
    while(infile >> tmp) {
        _fnames.push_back(tmp);
        std::cout << tmp << std::endl;
    }

    dims = nvinfer1::DimsNCHW(batchSize, channel, height, width);
    mInputCount1 = dims.n() * dims.c() * dims.h() * dims.w();
    cudaMalloc(&mDeviceInput1, mInputCount1 * sizeof(float));
};

EntropyCalibrator::~EntropyCalibrator()
{
    delete[] _batch;
    cudaFree(mDeviceInput1);
};

int EntropyCalibrator::getBatchSize() const { return dims.n();};

bool EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if(_cur_id + dims.n() >= _fnames.size()){
        return false;
    }

    for(int i = 0; i < dims.n(); i++){
        _cur_id+=1;
        std::string fname = _fnames[_cur_id];
        int width = dims.w();
        int height = dims.h();
        int channels = dims.c();
        unsigned char* img  = loadImageIO(fname.c_str(), &width, &height, &channels);


        for(int c=0; c<channels; c++){
            for(int h=0; h<height; h++){
                for(int w=0; w<width; w++){
                    int pix_id = h*w*c + w*c + c;
                    int row_id = i*h*w*c + c*h*w + h*w + w;
                    _batch[row_id] = float(img[pix_id]);
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

const void* EntropyCalibrator::readCalibrationCache(size_t& length)
{
    std::cout << "Reading from cache: "<< calibrationTableName()<<std::endl;
    mCalibrationCache.clear();
    std::ifstream input(calibrationTableName(), std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
};

void EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
};

std::string EntropyCalibrator::calibrationTableName()
{
    return "CalibrationTable";
}
