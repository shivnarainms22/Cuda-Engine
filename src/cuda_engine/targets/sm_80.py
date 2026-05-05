CAPS = {
    "name": "NVIDIA A100",
    "compute_capability": "8.0",
    "dtypes": ["fp32", "fp16", "bf16", "fp64", "int32", "int64", "uint8", "int8"],
    "tensor_cores": {
        "fp16": ["m16n8k16", "m16n8k32"],
        "bf16": ["m16n8k16", "m16n8k32"],
        "tf32": ["m16n8k8"],
    },
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_registers_per_thread": 255,
    "shared_memory_per_block_kb": 164,
    "recommendations": {
        "elementwise": {"threads_per_block": 256, "elements_per_thread": 4},
        "reductions": "warp-shuffle reduction followed by block shared-memory reduction",
    },
}
