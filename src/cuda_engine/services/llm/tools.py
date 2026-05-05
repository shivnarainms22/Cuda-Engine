from cuda_engine.services.llm.base import ToolSpec

COMPILE_KERNEL = ToolSpec(
    name="compile_kernel",
    description="Compile a CUDA source file for the requested target architecture.",
    input_schema={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Complete CUDA source for kernel.cu."},
            "target_arch": {
                "type": "string",
                "enum": ["sm_80", "sm_90", "sm_100", "sm_120"],
            },
            "extra_flags": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["src", "target_arch"],
        "additionalProperties": False,
    },
)

RUN_CORRECTNESS = ToolSpec(
    name="run_correctness",
    description="Run a compiled kernel against reference inputs and return numerical error stats.",
    input_schema={
        "type": "object",
        "properties": {
            "so_path": {"type": "string"},
            "shapes": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                },
            },
            "rtol": {"type": "number"},
            "atol": {"type": "number"},
        },
        "required": ["so_path", "shapes"],
        "additionalProperties": False,
    },
)

NSIGHT_PROFILE = ToolSpec(
    name="nsight_profile",
    description="Profile a compiled kernel with Nsight Compute and return key occupancy/memory metrics.",
    input_schema={
        "type": "object",
        "properties": {
            "so_path": {"type": "string"},
            "sample_shape": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
        },
        "required": ["so_path"],
        "additionalProperties": False,
    },
)

ALL_TOOLS = [COMPILE_KERNEL, RUN_CORRECTNESS, NSIGHT_PROFILE]
