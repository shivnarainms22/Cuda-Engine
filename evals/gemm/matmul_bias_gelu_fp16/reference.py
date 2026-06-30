def reference(a, b, bias):
    import torch
    return torch.nn.functional.gelu(a @ b + bias, approximate='tanh')
