def reference(x):
    import torch
    return torch.nn.functional.hardsigmoid(x)
