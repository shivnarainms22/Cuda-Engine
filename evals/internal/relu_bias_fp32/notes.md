# relu_bias_fp32

Common fused epilogue. Edge cases include negative and positive generated inputs. Expected speedup band: faster than eager, near torch.compile.
