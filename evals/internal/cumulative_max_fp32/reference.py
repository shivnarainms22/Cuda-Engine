def reference(x):
    import torch
    return torch.cummax(x, dim=-1).values
