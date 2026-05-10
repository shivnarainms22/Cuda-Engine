def reference(x):
    import torch
    return torch.exp(x.float() - x.float().max(dim=-1, keepdim=True).values).to(x.dtype)
