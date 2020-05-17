import torch

def export(model, input_shape, model_name="default_model", device = "cuda:0"):
    """Exports pytorch`s model to ONNX format

    model -- pytorch model
    input_shape -- tuple(h, w), length should be equal 2
    model_name -- string, optional, name of exported file
    device -- string, optional, 'cuda:0' / 'cpu' ..
    """
    assert len(input_shape) == 2, "Length os input_shape must equal 2"
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, *input_shape).to(device)

    dynamic_axes = {"input.1": [0]}
    torch.onnx.export(model, dummy_input, f"{model_name}.onnx", verbose=True, input_names=['input.1'],
                      dynamic_axes=dynamic_axes)
