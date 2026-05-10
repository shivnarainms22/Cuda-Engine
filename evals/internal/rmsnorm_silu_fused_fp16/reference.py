def reference(x):
    import torch
    y = x / torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + 1e-5).to(x.dtype)
    return y * torch.sigmoid(y)
