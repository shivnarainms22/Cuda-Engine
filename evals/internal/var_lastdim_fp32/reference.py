def reference(x):
    import torch
    return torch.var(x, dim=-1, unbiased=False)
