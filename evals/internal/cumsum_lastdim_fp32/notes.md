# cumsum_lastdim_fp32

Inclusive scan along the last dim of a 2-D tensor, fp32. Tests per-row prefix-sum. Expected band: reductions/scans can beat torch.compile.
