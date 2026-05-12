def reference(x):
    import torch
    rms = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + 1e-5)
    normalized = (x / rms.to(x.dtype))
    return normalized * torch.sigmoid(normalized.float()).to(x.dtype)
