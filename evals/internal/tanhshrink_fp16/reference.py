def reference(x):
    import torch
    return x - torch.tanh(x)
