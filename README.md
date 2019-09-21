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

model = TRTModel(
	"path to your engine file", #str 
	"input binding name", # str 
	[list on output binging names], # List[str]
	(256, 256, 256), # scale image preproc; Tuple[float]
	(0.5, 0.5, 0.5), #shift image preproc; Tuple[float] 
	max_batch_size #int
	)

import cv2

img = cv2.imread(img_path)
outputs = model.apply(img)
```
Convert model from ONNX into TRT Engine:
```
from tensorrt_models import import convertONNX, precisionType, deviceType, pixelFormat

convertONNX(
	"path to onnx", # str
	"path to file with paths for calib images", #str
	(256, 256, 256), # scale image preproc; Tuple[float]
	(0.5, 0.5, 0.5), #shift image preproc; Tuple[float]
        1, # maxBatch;int
        True, #allowGPUFallback 
        device = deviceType.DEVICE_GPU, 
        precision = precisionType.TYPE_INT8,
        format = pixelFormat.BGR)
```
Must know details:
>- Scale and Shift are used to make image preprocessing. Finally **float(image)/scale - shift** is fed into the network. The order of coeffs in this vectors (scale and shift) **must** correspont to input image format i.e. RGB. (None that openCV usually opens images as BGR).
>- Make sure that you **callibrate** your model in the **appropraite pixel format**. If you trained your net in RGB mode, the same format should be used during callibration.
>- To calibrate the model you need to create a file, containing paths to calibation images, and provide a path to this file.






