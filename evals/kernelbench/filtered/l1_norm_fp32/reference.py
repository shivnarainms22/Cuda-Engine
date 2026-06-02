def reference(x):
    import torch
    return x / torch.sum(torch.abs(x), dim=-1, keepdim=True)
