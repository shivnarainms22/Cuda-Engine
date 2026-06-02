def reference(x):
    import torch
    return torch.cumsum(x.flip(-1), dim=-1).flip(-1)
