def reference(x):
    import torch
    return torch.max(x, dim=-1).values
