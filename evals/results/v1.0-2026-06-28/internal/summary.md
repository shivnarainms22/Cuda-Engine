# CUDA Engine Eval Summary

Pass rate: 30/30

## M3 Metrics

- Pass rate: 30/30 (100.0%)
- Median speedup vs torch.compile: 1.04x
- P25 speedup vs torch.compile: 1.00x
- fast_1 kernels (>1.0x with measured baseline): 24/30
- baseline_failed (not counted in fast_1): 0/30
- Below target kernels (with measured baseline): 6/30

## Failure Breakdown

- External/API failures: 0
- Stage/kernel failures: 0
- Runner failures: 0

| Kernel | Status | Speedup vs torch.compile | Regression | Failure kind |
|---|---|---:|---|---|
| add_relu_fp32 | PASS | 1.00 |  |  |
| argmax_fp32 | PASS | 1.06 |  |  |
| bias_gelu_fp16 | PASS | 1.03 |  |  |
| clamp_fp32 | PASS | 1.01 |  |  |
| cumulative_max_fp32 | PASS | 1.45 |  |  |
| dropout_fp16 | PASS | 1.08 |  |  |
| geglu_fp16 | PASS | 1.02 |  |  |
| gelu_fp16 | PASS | 0.67 |  |  |
| l2_norm_fp32 | PASS | 1.13 |  |  |
| layernorm_fp16 | PASS | 1.05 |  |  |
| layernorm_silu_fused_fp16 | PASS | 1.08 |  |  |
| masked_mean_fp16 | PASS | 2.57 |  |  |
| max_lastdim_fp32 | PASS | 1.08 |  |  |
| mean_lastdim_fp32 | PASS | 1.10 |  |  |
| min_lastdim_fp32 | PASS | 1.10 |  |  |
| prefix_sum_fp32 | PASS | 0.95 |  |  |
| relu_bias_fp32 | PASS | 0.98 |  |  |
| rms_norm_fp16 | PASS | 0.75 |  |  |
| rmsnorm_silu_fused_fp16 | PASS | 1.04 |  |  |
| scalar_multiply_fp32 | PASS | 1.01 |  |  |
| segment_sum_fp32 | PASS | 1.01 |  |  |
| sigmoid_mul_fp16 | PASS | 1.01 |  |  |
| silu_fp16 | PASS | 1.07 |  |  |
| softmax_lastdim_fp16 | PASS | 1.33 |  |  |
| softmax_numerator_fp16 | PASS | 1.04 |  |  |
| sum_reduction_fp32 | PASS | 1.10 |  |  |
| swiglu_fp16 | PASS | 1.00 |  |  |
| tanh_add_fp32 | PASS | 0.99 |  |  |
| topk_fp32 | PASS | 12.47 |  |  |
| vector_add_fp32 | PASS | 0.98 |  |  |
