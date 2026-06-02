def reference(x):
    import torch
    return torch.nn.functional.softsign(x)
