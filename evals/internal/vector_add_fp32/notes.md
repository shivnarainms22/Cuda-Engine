# vector_add_fp32

Baseline elementwise memory-bandwidth kernel. Edge cases cover small, power-of-two, and tail sizes. Expected speedup band: near torch.compile to modestly faster for large N.
