def reference(x):
    import torch
    return torch.topk(x, k=1, dim=-1).values.squeeze(-1)
