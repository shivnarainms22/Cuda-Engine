def reference(x, mask):
    import torch
    return torch.cumsum(x * mask, dim=-1)
