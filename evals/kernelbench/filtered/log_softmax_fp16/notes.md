# log_softmax_fp16

Row-wise fused reduction (max + sum-exp) followed by elementwise. Source: KernelBench level1/24_LogSoftmax.py (MIT). Numerically stable log-softmax expected.
