def reference(x):
    import torch
    return torch.log_softmax(x, dim=-1)
