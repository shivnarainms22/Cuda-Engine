def reference(x):
    import torch
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    return ((x.float() - mean) / torch.sqrt(var + 1e-5)).to(x.dtype)
