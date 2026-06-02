def reference(predictions, targets):
    import torch
    return torch.mean((predictions - targets) ** 2)
