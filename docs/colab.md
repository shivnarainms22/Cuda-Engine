# Colab Runbook

This runbook is the tested workflow for running `cuda-engine` in Google Colab with a real Anthropic API key and a CUDA runtime.

## 1. Start A GPU Runtime

In Colab:

1. Open `Runtime`.
2. Select `Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.
4. Save.

Verify CUDA is visible:

```python
!nvidia-smi
```

```python
!which nvcc
```

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

Expected:

```text
True
```

If `torch.cuda.is_available()` is `False`, restart with a GPU runtime before continuing.

## 2. Clone Or Update The Repo

For a fresh Colab session:

```python
%cd /content
```

```python
!git clone https://github.com/shivnarainms22/Cuda-Engine.git
```

```python
%cd /content/Cuda-Engine
```

```python
!git checkout m2/stage1-interview
```

For an existing Colab checkout:

```python
%cd /content/Cuda-Engine
```

```python
!git checkout m2/stage1-interview
```

```python
!git pull
```

## 3. Install The Package

Install in editable mode with dev dependencies:

```python
%cd /content/Cuda-Engine
```

```python
!python -m pip install -e ".[dev]"
```

Verify imports:

```python
from cuda_engine import synthesize
print(synthesize)
```

If this raises `ModuleNotFoundError: No module named 'cuda_engine'`, run the editable install cell again from `/content/Cuda-Engine`.

## 4. Set The Anthropic API Key

Use `getpass()` and paste only the key when prompted.

```python
import os
from getpass import getpass

os.environ["ANTHROPIC_API_KEY"] = getpass("ANTHROPIC_API_KEY: ")
```

Do not paste the key into the prompt string. This is wrong:

```python
os.environ["ANTHROPIC_API_KEY"] = getpass("ANTHROPIC_API_KEY: sk-ant-...")
```

That makes the notebook appear stuck waiting for hidden input.

Verify the variable exists without printing the secret:

```python
import os
print("ANTHROPIC_API_KEY" in os.environ)
```

Expected:

```text
True
```

## 5. Run Local Unit Tests

```python
%cd /content/Cuda-Engine
```

```python
!pytest tests/unit -v
```

Expected:

```text
passed
```

## 6. Run GPU Integration Smoke Tests

Run the known-good Torch custom op integration:

```python
!pytest tests/integration/test_run_kernel_custom_op.py -v -m integration -s
```

Run the real LLM plus compile end-to-end integration:

```python
!pytest tests/integration/test_e2e_vector_add.py -v -m integration -s
```

Expected:

```text
passed
```

These tests require:

- `ANTHROPIC_API_KEY` in the environment.
- `nvcc` available.
- `torch.cuda.is_available()` returning `True`.

## 7. Inspect The Latest Report

Use the CLI convenience command:

```python
!cuda-engine latest-report .test_artifacts/runs
```

To inspect the raw JSON:

```python
!cat $(find .test_artifacts/runs -name report.json | sort | tail -1)
```

The report includes:

- top-level pass/fail status
- failed stage and reason, if any
- stage traces
- LLM token totals
- correctness report
- performance report
- artifact directory

## 8. Run A Manual Synthesis Call

```python
from cuda_engine import SynthesisConfig, synthesize

result = synthesize(
    "Generate a CUDA kernel for vector addition: out = x + y for fp32 tensors.",
    lambda x, y: x + y,
    "sm_80",
    config=SynthesisConfig(artifact_root=".test_artifacts/runs"),
)

print(result.passed)
print(result.run_id)
print(result.artifacts_dir)
```

Then inspect the latest report:

```python
!cuda-engine latest-report .test_artifacts/runs
```

## Known Failures And Fixes

### `ModuleNotFoundError: No module named 'cuda_engine'`

Cause: the package is not installed in the active Colab Python environment.

Fix:

```python
%cd /content/Cuda-Engine
```

```python
!python -m pip install -e ".[dev]"
```

### API Key Cell Appears To Hang

Cause: the key was pasted into the `getpass()` prompt string instead of into the hidden input prompt.

Fix:

```python
import os
from getpass import getpass

os.environ["ANTHROPIC_API_KEY"] = getpass("ANTHROPIC_API_KEY: ")
```

Paste the key after Colab prompts for input.

### Anthropic Error: `tools: Input should be a valid array`

Cause: older code sent `tools=None` to Anthropic.

Fix: pull the latest branch. The Anthropic client now omits the `tools` field when there are no tools.

```python
%cd /content/Cuda-Engine
```

```python
!git pull
```

### Anthropic Error: `temperature is deprecated for this model`

Cause: older code sent a default temperature to models that reject it.

Fix: pull the latest branch. The LLM client default temperature is `None`, and Anthropic only receives temperature when explicitly configured.

### Stage 1 Fails On `layout_hint: contiguous`

Cause: real model output used `contiguous`, while the schema accepts `row_major`, `col_major`, or `any`.

Fix: pull the latest branch. Stage 1 now normalizes `contiguous`, `c_contiguous`, and `strided_contiguous` to `row_major`.

### Compile Fails With `torch/extension.h: No such file or directory`

Cause: custom op compilation needs Torch and Python include/library flags.

Fix: pull the latest branch. `LocalGPURunner.compile()` adds the Torch/Python compile and link flags when the source uses Torch extension/custom-op APIs.

### Runtime ImportError: `PyInit_cuda_engine_generated_kernel`

Cause: Torch custom op `.so` files are not normal Python extension modules.

Fix: pull the latest branch. The child runner falls back from Python module import to `torch.ops.load_library()`.

### Correctness Fails With CPU Backend Error

Typical error:

```text
Could not run 'cuda_engine::forward' with arguments from the 'CPU' backend
```

Cause: Stage 3 generated CPU tensors while the custom op was registered for CUDA.

Fix: pull the latest branch. Stage 3 now creates CUDA tensors when CUDA is available.

## Minimal End-To-End Cell List

```python
%cd /content
```

```python
!git clone https://github.com/shivnarainms22/Cuda-Engine.git
```

```python
%cd /content/Cuda-Engine
```

```python
!git checkout m2/stage1-interview
```

```python
!python -m pip install -e ".[dev]"
```

```python
import os
from getpass import getpass

os.environ["ANTHROPIC_API_KEY"] = getpass("ANTHROPIC_API_KEY: ")
```

```python
!pytest tests/unit -v
```

```python
!pytest tests/integration/test_run_kernel_custom_op.py -v -m integration -s
```

```python
!pytest tests/integration/test_e2e_vector_add.py -v -m integration -s
```

```python
!cuda-engine latest-report .test_artifacts/runs
```
