def reference(x):
    import torch
    return torch.clamp(x, -1.0, 1.0)
