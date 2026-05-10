def reference(x, bias):
    import torch
    return torch.nn.functional.gelu(x + bias, approximate='tanh')
