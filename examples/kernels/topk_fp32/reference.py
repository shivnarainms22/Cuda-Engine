def reference(x, k):
    import torch
    return torch.topk(x, k, dim=-1, largest=True, sorted=True)
