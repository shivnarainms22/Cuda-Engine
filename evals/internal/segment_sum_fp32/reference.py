def reference(x):
    width = 4
    trimmed = x[..., : (x.shape[-1] // width) * width]
    return trimmed.reshape(*trimmed.shape[:-1], -1, width).sum(dim=-1)
