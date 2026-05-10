def reference(x, gate):
    import torch
    return x * torch.nn.functional.gelu(gate, approximate='tanh')
