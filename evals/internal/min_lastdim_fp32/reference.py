def reference(x):
    import torch
    return torch.min(x, dim=-1).values
