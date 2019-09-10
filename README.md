TensorRT inference in C++
===================
----------
This project is aimed at providing fast inference for NN with tensorRT through its C++ API without any need of C++ programming. Use your lovely python.

Instructions
-------------
> - git clone
> - cd <project folder>
> - pip install .

Then you can:
```
from OpenposeTensorRT import precisionType, Openpose

model = Openpose("path to your engine file", "input binding name", [list on output binging names], max_batch_size, precisionType)

import numpy as np
import cv2

img = numpy(cv2.imread(img_path), order = "C")
outputs = model.apply(img)
```


