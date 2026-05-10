def reference(x):
    import torch
    return torch.where(x > 0, x * 2, torch.zeros_like(x))
