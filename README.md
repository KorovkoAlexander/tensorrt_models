TensorRT inference in Python
===================
----------
This project is aimed at providing fast inference for NN with tensorRT through its C++ API without any need of C++ programming. Use your lovely python.

Build Instructions
-------------
#### Windows prebuild instructions:

> 1) install Visual Studio
> 2) THEN install CUDA (make sure that VS version satisfies cuda requirements

#### Build:

> 1) Download TensorRT library and extract it.
> 2) git clone --recursive https://github.com/KorovkoAlexander/tensorrt_models
> 3) cd tensorrt_models
> 4) Open CmakeLists.txt and Change TensorRT include and lib paths.
> 5) pip install .

Then you can:
```
from tensorrt_models import TRTModel

model = TRTModel("path to your engine file", "input binding name", [list on output binging names], max_batch_size)

import cv2

img = cv2.imread(img_path)
outputs = model.apply(img)
```
Convert model from ONNX into TRT Engine:
```
from tensorrt_models import import convertONNX, precisionType, deviceType

convertONNX("path to onnx", 
	    "path to file with paths for calib images", 
            maxBatch, 
            device = deviceType.DEVICE_GPU, 
            precision = precisionType.TYPE_INT8)
```



