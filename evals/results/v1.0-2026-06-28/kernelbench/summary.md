# CUDA Engine Eval Summary

Pass rate: 12/12

## M3 Metrics

- Pass rate: 12/12 (100.0%)
- Median speedup vs torch.compile: 1.05x
- P25 speedup vs torch.compile: 1.03x
- fast_1 kernels (>1.0x with measured baseline): 11/12
- baseline_failed (not counted in fast_1): 0/12
- Below target kernels (with measured baseline): 1/12

## Failure Breakdown

- External/API failures: 0
- Stage/kernel failures: 0
- Runner failures: 0

| Kernel | Status | Speedup vs torch.compile | Regression | Failure kind |
|---|---|---:|---|---|
| argmin_fp32 | PASS | 1.05 |  |  |
| elu_fp16 | PASS | 1.05 |  |  |
| frobenius_norm_fp32 | PASS | 1.17 |  |  |
| l1_norm_fp32 | PASS | 0.94 |  |  |
| leaky_relu_fp16 | PASS | 1.12 |  |  |
| log_softmax_fp16 | PASS | 1.31 |  |  |
| masked_cumsum_fp32 | PASS | 1.03 |  |  |
| mingpt_gelu_fp16 | PASS | 1.03 |  |  |
| mse_loss_fp32 | PASS | 1.00 |  |  |
| reverse_cumsum_fp32 | PASS | 1.86 |  |  |
| softplus_fp16 | PASS | 1.12 |  |  |
| softsign_fp16 | PASS | 1.01 |  |  |
