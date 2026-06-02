def reference(x):
    import torch
    return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
