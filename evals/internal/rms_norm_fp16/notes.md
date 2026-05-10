# rms_norm_fp16

M2 normalization kernel with last-dim reduction and fp32 accumulation. Edge cases vary rows and hidden size. Expected speedup band: can beat eager; torch.compile parity acceptable.
