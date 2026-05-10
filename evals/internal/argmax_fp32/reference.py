def reference(x):
    import torch
    return torch.argmax(x, dim=-1)
