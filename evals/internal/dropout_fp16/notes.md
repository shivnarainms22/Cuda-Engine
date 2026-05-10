# dropout_fp16

Deterministic dropout surrogate avoids RNG while testing mask + scale. Expected speedup band: near torch.compile.
