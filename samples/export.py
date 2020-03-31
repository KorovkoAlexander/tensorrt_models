import torch

def export(model, input_shape, model_name="default_model", device = "cuda:0"):
    """Exports pytorch`s model to ONNX format

    model -- pytorch model
    input_shape -- tuple(-1, c, h, w), length should be equal 4
    model_name -- string, optional, name of exported file
    device -- string, optional, 'cuda:0' / 'cpu' ..
    """
    assert len(input_shape) == 4, "Length os input_shape must equal 4"
    assert input_shape[0] == -1, "Batchsize must equal -1"
    model = model.to(device)
    model.eval()
    dummy = torch.rand(input_shape).to(device)

    input_names = ["input.1"]
    dynamic_axes = dict(zip(input_names, [{0:'batch_size'} for i in range(len(input_names))]))
    torch.onnx.export(model, dummy, f"{model_name}.onnx", verbose=False,
                      opset_version=11, input_names=['input.1'],
                      dynamic_axes=dynamic_axes)


def export_onnx(model, input_shape, model_name="default_model", device = "cuda:0"):
    """Exports pytorch`s model to ONNX format

    model -- pytorch model
    input_shape -- tuple(max_batch_size, c, h, w), length should be equal 4
    model_name -- string, optional, name of exported file
    device -- string, optional, 'cuda:0' / 'cpu' ..
    """
    assert len(input_shape) == 4, "Length os input_shape must equal 4"
    assert input_shape[0] > 0, "Batchsize must be greater then 0"
    model = model.to(device)
    model.eval()
    dummy = torch.rand(input_shape).to(device)

    torch.onnx.export(model, dummy, f"{model_name}.onnx", verbose=False,
                      opset_version=11)