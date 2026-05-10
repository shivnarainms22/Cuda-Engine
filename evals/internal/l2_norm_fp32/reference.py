def reference(x):
    import torch
    return torch.sqrt(torch.sum(x * x, dim=-1))
