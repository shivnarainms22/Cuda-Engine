def reference(x):
    import torch
    return torch.argmin(x, dim=-1)
