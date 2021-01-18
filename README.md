TensorRT inference in Python
===================
----------
This project is aimed at providing fast inference for NN with tensorRT through its C++ API without any need of C++ programming. Use your lovely python.

#### Examples
[GoogleDrive](https://drive.google.com/open?id=1mdh9E0s5SNf48scuUvrheff345cOfaRf)

#### Important
Currently CUDA 10.2, TensorRT 7.1.3.4 and TensorRT 7.2.1.6 is supported.
Also CUDA 11.1 and TensorRT 7.2.2

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

Prebuild packages
----------------
[Prebuild](https://github.com/KorovkoAlexander/tensorrt_models/issues/1#issue-591786674)

Then you can:
```
from tensorrt_models import TRTModel

model = TRTModel(
	model_path = "path to your engine file", #str 
	device = 0, #on which GPU to run #int
	logs_path = "path to logs file" #str
	)

import cv2

img = cv2.imread(img_path)
img1 = cv2.imread(img_path_1)
img2 = cv2.imread(img_path_2)

batch = np.stack([img1, img2, img3]).transpose((0, 3, 1, 2)) # shape = (b, c, h, w)
outputs = model.apply(batch)
```
Convert model from ONNX into TRT Engine:
```
from tensorrt_models import import convertONNX, precisionType, deviceType, pixelFormat

convertONNX(
	modelFile = "path to onnx", # str
	file_list = "path to file with paths for calib images", #str
	scale = (58.395, 57.12 , 57.375), # scale image preproc; Tuple[float]
	shift = (123.675, 116.28 , 103.53), #shift image preproc; Tuple[float]
    max_batch_size = 1, # maxBatch;int
    allowGPUFallback = True, #allowGPUFallback 
    device = deviceType.DEVICE_GPU, 
    precision = precisionType.TYPE_INT8,
    format = pixelFormat.RGB,
    logs_path = "path to logs file" #str)
```
Must know details:
>- Make sure that you **callibrate** your model in the **appropraite pixel format**. If you trained your net in RGB mode, the same format should be used during callibration.
>- Scale and Shift are used to make image preprocessing during calibration. Finally **(image - shift)/scale** is fed into the network. The order of coeffs in this vectors (scale and shift) **must** correspont to input image format i.e. RGB. (None that openCV usually opens images as BGR).
>- To calibrate the model you need to create a file, containing paths to calibation images, and provide a path to this file.






