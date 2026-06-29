# gelu_erf_fp16

Exact (erf-based) GELU, fp16. Complements the tanh-approx gelu; tests the erf/transcendental path. Expected band: parity to modestly slower than torch.compile.
