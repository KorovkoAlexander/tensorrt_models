import torch
import numpy as np
from torchvision.models import resnet18

from tensorrt_models import convertONNX, TRTModel

model = resnet18(pretrained=True)
model.eval()

model = model.cuda()
d = torch.ones(1, 3, 224, 224).cuda()

out = model(d)
print(out[0][:10])

da = {"input": {0:"batch_size"}}
torch.onnx.export(model, d, "resnet.onnx", verbose=False, input_names=["input"], opset_version=9, dynamic_axes=da)


convertONNX("resnet.onnx", logs_path="logs.txt")

modeltrt = TRTModel("resnet.engine")

x = np.ones((1, 3, 224, 224), dtype=np.float32)

out2 = modeltrt.apply(x)

print(out2[0][:10])

