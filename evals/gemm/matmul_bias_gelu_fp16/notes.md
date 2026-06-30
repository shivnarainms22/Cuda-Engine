# matmul_bias_gelu_fp16

FUSED GEMM epilogue — the eventual VALUE target for v2.0. torch does matmul (cuBLAS) then a separate bias+gelu kernel, paying an extra N x N read+write; a fused kernel saves that round-trip, so this is where custom codegen can actually BEAT the baseline even with a merely-decent inner GEMM.
