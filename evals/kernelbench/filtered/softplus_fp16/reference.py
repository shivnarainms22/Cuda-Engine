def reference(x):
    import torch
    return torch.nn.functional.softplus(x)
