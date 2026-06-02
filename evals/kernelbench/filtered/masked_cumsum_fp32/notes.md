# masked_cumsum_fp32

Elementwise multiply followed by a prefix-sum scan over the last dimension; mask is modeled as a second float tensor for the eval harness. Source: KernelBench level1/93_masked_cumsum.py (MIT, dim adapted to last dim).
