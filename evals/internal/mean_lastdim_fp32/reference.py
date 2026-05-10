def reference(x):
    import torch
    return torch.mean(x, dim=-1)
