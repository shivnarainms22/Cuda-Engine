def reference(x):
    import torch
    return torch.cumsum(x, dim=-1)
