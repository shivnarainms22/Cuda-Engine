# matmul_fp16

Square fp16 GEMM (the core correctness de-risk for v2.0). Tests whether the engine can generate a correct shared-memory-tiled GEMM. Baseline = torch.matmul (cuBLAS); we expect to LOSE on the bare GEMM (cuBLAS is near-peak) — the goal here is CORRECTNESS + a reasonable fraction of cuBLAS, not beating it.
