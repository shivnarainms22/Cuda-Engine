def reference(x, gate):
    import torch
    return x * torch.sigmoid(gate)
