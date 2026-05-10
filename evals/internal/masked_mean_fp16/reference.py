def reference(x):
    import torch
    mask = x > 0
    numer = torch.where(mask, x.float(), torch.zeros_like(x.float())).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(1).float()
    return (numer / denom).to(x.dtype)
