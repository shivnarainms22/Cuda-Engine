def reference(x, bias):
    import torch
    return torch.relu(x + bias)
