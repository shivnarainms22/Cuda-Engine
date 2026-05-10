def reference(x):
    import torch
    return torch.softmax(x, dim=-1)
