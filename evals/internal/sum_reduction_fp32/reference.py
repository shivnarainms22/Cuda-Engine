def reference(x):
    import torch
    return torch.sum(x, dim=-1)
