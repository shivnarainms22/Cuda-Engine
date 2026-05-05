# CUDA Synthesis Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library + CLI that turns a plain-English prompt + a PyTorch reference function into a verified, benchmarked, annotated CUDA kernel via a 5-stage Claude-driven agent loop, hitting the v1 perf bar (≥95% functional / median ≥1.0× torch.compile / ≥30% fast_1) on the internal regression set.

**Architecture:** Monolithic Python package with three layers — public API → Orchestrator + 5 Stages (pure logic) → Services (LLMClient, GPURunner, ArtifactStore — interfaces with one impl each in v1). Anthropic-only LLM backend with prompt caching + tool use. PyTorch reference function is the numerical-correctness oracle. Stage 3 is a hard gate; Stage 4 is a soft gate. Subprocess-isolated GPU work. Full provenance written to flat-file run dirs.

**Tech Stack:** Python 3.11+, `anthropic` (latest with prompt cache), `torch` 2.4+ (`cpp_extension.load_inline`), `pydantic` v2, `typer` (CLI), `pytest` + `pytest-asyncio`, `ruff`, `mypy`. CUDA toolkit 12.x. Streamlit only in `examples/`. Dev hardware: A100 via Colab Pro.

**Spec:** `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md`

**Skills referenced:**
- @superpowers:test-driven-development — every task follows red→green→refactor.
- @superpowers:verification-before-completion — checkpoint passes only with evidence (real command output).
- @superpowers:systematic-debugging — when a step fails, diagnose root cause; don't paper over.

---

## File Structure (locked)

```
cuda-engine/
├── pyproject.toml                         # PEP 621, deps + dev-deps + scripts
├── ruff.toml                              # lint config
├── README.md                              # placeholder until M4
├── LICENSE                                # MIT
├── .gitignore                             # python + caches + run dirs
├── .github/workflows/
│   ├── pr.yml                             # unit + lint + types (M0)
│   ├── nightly.yml                        # integration (M1)
│   └── eval.yml                           # full eval (M4)
│
├── src/cuda_engine/
│   ├── __init__.py                        # exports synthesize, SynthesisResult, SynthesisConfig
│   ├── api.py                             # synthesize() public entry point (M0 stub, M1 real)
│   ├── cli.py                             # Typer CLI (M0 skeleton, M3 real, M4 final)
│   ├── config.py                          # SynthesisConfig pydantic model
│   ├── orchestrator.py                    # Orchestrator class (M0 stub, M2 hard gate, M3 escalation)
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py                        # Stage ABC + retry helper
│   │   ├── interview.py                   # Stage 1 (M2)
│   │   ├── codegen.py                     # Stage 2 (M1)
│   │   ├── correctness.py                 # Stage 3 (M2 hard gate)
│   │   ├── performance.py                 # Stage 4 (M2 stub, M3 real)
│   │   └── polish.py                      # Stage 5 (M2)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── base.py                    # LLMClient ABC
│   │   │   ├── mocks.py                   # MockLLMClient
│   │   │   ├── anthropic.py               # AnthropicClient (M1)
│   │   │   └── tools.py                   # tool schemas
│   │   ├── gpu/
│   │   │   ├── base.py                    # GPURunner ABC
│   │   │   ├── mocks.py                   # MockGPURunner
│   │   │   └── local.py                   # LocalGPURunner (M1 compile, M2 run, M3 profile)
│   │   └── store/
│   │       ├── base.py                    # ArtifactStore ABC
│   │       ├── mocks.py                   # InMemoryStore
│   │       └── local_dir.py               # LocalDirStore (M1)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spec.py                        # KernelSpec
│   │   ├── artifact.py                    # KernelArtifact
│   │   └── reports.py                     # CorrectnessReport, PerformanceReport, SynthesisReport
│   │
│   ├── prompts/                           # prompt templates (markdown)
│   │   ├── interview.md                   # M2
│   │   ├── codegen.md                     # M1
│   │   ├── correctness_fix.md             # M2
│   │   ├── perf_fix.md                    # M3
│   │   └── polish.md                      # M2
│   │
│   └── targets/
│       ├── __init__.py
│       ├── sm_80.py                       # A100 — full content (M1)
│       ├── sm_90.py                       # H100 — placeholder (M2)
│       └── sm_100.py                      # B200 — placeholder (M2)
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_api.py
│   │   ├── test_orchestrator.py
│   │   ├── stages/
│   │   │   ├── test_interview.py
│   │   │   ├── test_codegen.py
│   │   │   ├── test_correctness.py
│   │   │   ├── test_performance.py
│   │   │   └── test_polish.py
│   │   ├── services/
│   │   │   ├── llm/test_anthropic.py
│   │   │   ├── gpu/test_local.py
│   │   │   └── store/test_local_dir.py
│   │   └── models/
│   │       └── test_models.py
│   └── integration/
│       ├── test_e2e_vector_add.py         # M1
│       ├── test_e2e_rmsnorm.py            # M2
│       └── test_e2e_reduction_sum.py      # M2
│
├── evals/
│   ├── internal/                          # ~30 kernel dirs (built up M2→M3)
│   │   └── <kernel_name>/
│   │       ├── prompt.txt
│   │       ├── reference.py
│   │       ├── shapes.yaml
│   │       └── notes.md
│   ├── kernelbench/
│   │   ├── README.md
│   │   └── filter.py                      # M4
│   ├── runner.py                          # M3
│   └── results/<date>/                    # auto-generated
│
├── examples/
│   ├── notebook.ipynb                     # Colab quickstart (M4)
│   ├── web_demo.py                        # Streamlit (M4)
│   └── kernels/                           # 5–10 worked examples (M4)
│
└── docs/                                  # already exists from brainstorming
    ├── brainstorming-notes.md
    └── superpowers/
        ├── specs/2026-04-26-cuda-synthesis-engine-design.md
        └── plans/2026-04-26-cuda-synthesis-engine-plan.md      # this file
```

---

## Chunk 1 — Milestone 0: Skeleton

**Goal:** Repo bootstrapped, all interfaces defined, mocks in place, end-to-end mock pipeline runs in <30s on a laptop with no LLM/GPU. PR CI green. Establishes the patterns every subsequent milestone reuses.

**Definition of done = M0 checkpoint from the design:**
- `pip install -e .` works
- `pytest tests/unit` green with ≥1 test per module
- `ruff check .` and `mypy src/` clean
- `synthesize("noop", lambda x: x, "sm_80")` returns `SynthesisResult.ok` end-to-end through mocks
- PR CI workflow runs on push

### Task 1.1: Initial repo + pyproject + .gitignore + LICENSE

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md` (placeholder), `ruff.toml`
- Create: `src/cuda_engine/__init__.py` (empty for now)

- [ ] **Step 1: Init git and create directory layout**

```bash
cd "D:/Cuda Engine"
git init
mkdir -p src/cuda_engine tests/unit tests/integration
touch src/cuda_engine/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "cuda-engine"
version = "0.0.1"
description = "Plain-English -> verified CUDA kernels via Claude-driven agent loop"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "anthropic>=0.40",
    "torch>=2.4",
    "pydantic>=2.7",
    "typer>=0.12",
    "pyyaml>=6",
    "rich>=13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5",
    "ruff>=0.6",
    "mypy>=1.10",
]
demo = ["streamlit>=1.36"]

[project.scripts]
cuda-engine = "cuda_engine.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/cuda_engine"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
    "integration: real LLM + real GPU (slow, costs money)",
]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/cuda_engine"]
```

- [ ] **Step 3: Write `ruff.toml`**

```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.coverage
htmlcov/
.cache/
runs/                       # local run dirs if user opts in
*.so
*.cubin
*.ptx
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 5: Write `README.md` placeholder**

```markdown
# cuda-engine

Plain English → verified CUDA kernels.

Status: under construction. See `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md`.
```

- [ ] **Step 6: Write `LICENSE` (MIT)** — standard MIT text, copyright 2026 Shivnarain.

- [ ] **Step 7: Verify install works**

Run:
```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows bash
pip install -e ".[dev]"
```

Expected: install completes, no errors. `python -c "import cuda_engine"` succeeds.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml ruff.toml .gitignore LICENSE README.md src/cuda_engine/__init__.py
git commit -m "chore: bootstrap project (pyproject, ruff, mypy, license)"
```

---

### Task 1.2: Data models — `KernelSpec`, `KernelArtifact`, reports

**Files:**
- Create: `src/cuda_engine/models/__init__.py`
- Create: `src/cuda_engine/models/spec.py`
- Create: `src/cuda_engine/models/artifact.py`
- Create: `src/cuda_engine/models/reports.py`
- Test: `tests/unit/models/__init__.py`, `tests/unit/models/test_models.py`

- [ ] **Step 1: Write failing tests for `KernelSpec`**

`tests/unit/models/test_models.py`:
```python
from cuda_engine.models.spec import KernelSpec, TensorArg, OptimizationPriority


def test_kernel_spec_minimal_round_trip():
    spec = KernelSpec(
        name="vector_add",
        target_arch="sm_80",
        inputs=[
            TensorArg(name="x", dtype="fp32", shape=("N",)),
            TensorArg(name="y", dtype="fp32", shape=("N",)),
        ],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
        precision_tolerance={"rtol": 1e-5, "atol": 1e-6},
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )
    j = spec.model_dump_json()
    parsed = KernelSpec.model_validate_json(j)
    assert parsed == spec


def test_kernel_spec_rejects_unknown_dtype():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelSpec(
            name="bad",
            target_arch="sm_80",
            inputs=[TensorArg(name="x", dtype="float37", shape=("N",))],
            outputs=[TensorArg(name="o", dtype="fp32", shape=("N",))],
            precision_tolerance={"rtol": 1e-5, "atol": 1e-6},
            optimization_priority=OptimizationPriority.LATENCY,
        )
```

- [ ] **Step 2: Run tests, observe failure**

```bash
pytest tests/unit/models/test_models.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `models/spec.py`**

```python
"""KernelSpec — the frozen contract output by Stage 1."""
from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

DType = Literal["fp32", "fp16", "bf16", "fp64", "int32", "int64", "uint8", "int8"]
TargetArch = Literal["sm_80", "sm_90", "sm_100", "sm_120"]


class OptimizationPriority(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


class TensorArg(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    dtype: DType
    shape: tuple[str, ...] = Field(description="Symbolic shape, e.g. ('B','S','D')")
    layout_hint: Literal["row_major", "col_major", "any"] = "any"


class PrecisionTolerance(BaseModel):
    model_config = ConfigDict(frozen=True)
    rtol: float = 1e-3
    atol: float = 1e-3


class KernelSpec(BaseModel):
    """Frozen after Stage 1; downstream stages must not mutate."""
    model_config = ConfigDict(frozen=True)
    name: str
    target_arch: TargetArch
    inputs: list[TensorArg]
    outputs: list[TensorArg]
    precision_tolerance: PrecisionTolerance | dict[str, float]
    optimization_priority: OptimizationPriority
    notes: str = ""
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/unit/models/test_models.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Add tests + impl for `KernelArtifact` and reports**

Add to `test_models.py`:
```python
from pathlib import Path
from cuda_engine.models.artifact import KernelArtifact
from cuda_engine.models.reports import CorrectnessReport, PerformanceReport, SynthesisReport


def test_kernel_artifact_round_trip(tmp_path: Path):
    art = KernelArtifact(
        kernel_cu_path=tmp_path / "kernel.cu",
        kernel_so_path=tmp_path / "kernel.so",
        compile_log="ok",
        ptx_size_bytes=1234,
    )
    j = art.model_dump_json()
    parsed = KernelArtifact.model_validate_json(j)
    assert parsed.kernel_cu_path == art.kernel_cu_path


def test_correctness_report_passed_property():
    r = CorrectnessReport(
        passed=True,
        max_abs_err=1e-6,
        max_rel_err=1e-6,
        shapes_tested=[(128,), (1024,)],
        failing_inputs=[],
    )
    assert r.passed is True


def test_synthesis_report_aggregates():
    rep = SynthesisReport(
        run_id="abc123",
        spec_name="vector_add",
        stages_executed=["interview", "codegen", "correctness", "performance", "polish"],
        total_llm_tokens_in=1000,
        total_llm_tokens_out=400,
        total_cost_usd=0.05,
        wall_time_seconds=42.0,
    )
    assert rep.run_id == "abc123"
```

`models/artifact.py`:
```python
from pathlib import Path
from pydantic import BaseModel, ConfigDict

class KernelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)
    kernel_cu_path: Path
    kernel_so_path: Path | None = None
    compile_log: str = ""
    ptx_size_bytes: int = 0
```

`models/reports.py`:
```python
from typing import Any
from pydantic import BaseModel

class CorrectnessReport(BaseModel):
    passed: bool
    max_abs_err: float
    max_rel_err: float
    shapes_tested: list[tuple[int, ...]]
    failing_inputs: list[dict[str, Any]] = []

class PerformanceReport(BaseModel):
    speedup_vs_reference: float
    speedup_vs_torch_compile: float
    achieved_tflops: float | None = None
    achieved_gbps: float | None = None
    occupancy: float | None = None
    regs_per_thread: int | None = None
    spill_bytes: int = 0
    below_target: bool = False

class StageTrace(BaseModel):
    stage_name: str
    attempts: int
    succeeded: bool
    model_used: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_seconds: float = 0.0

class SynthesisReport(BaseModel):
    run_id: str
    spec_name: str
    stages_executed: list[str]
    stage_traces: list[StageTrace] = []
    total_llm_tokens_in: int = 0
    total_llm_tokens_out: int = 0
    total_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    warnings: list[str] = []
```

`models/__init__.py`:
```python
from cuda_engine.models.spec import KernelSpec, TensorArg, PrecisionTolerance, OptimizationPriority
from cuda_engine.models.artifact import KernelArtifact
from cuda_engine.models.reports import CorrectnessReport, PerformanceReport, SynthesisReport, StageTrace

__all__ = [
    "KernelSpec", "TensorArg", "PrecisionTolerance", "OptimizationPriority",
    "KernelArtifact",
    "CorrectnessReport", "PerformanceReport", "SynthesisReport", "StageTrace",
]
```

- [ ] **Step 6: Run all model tests**

```bash
pytest tests/unit/models/ -v
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/cuda_engine/models tests/unit/models
git commit -m "feat(models): KernelSpec, KernelArtifact, reports with pydantic v2"
```

---

### Task 1.3: `SynthesisConfig` and `SynthesisResult`

**Files:**
- Create: `src/cuda_engine/config.py`
- Modify: `src/cuda_engine/__init__.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_config.py`:
```python
from cuda_engine.config import SynthesisConfig, RetryBudgets


def test_default_retry_budgets():
    cfg = SynthesisConfig()
    assert cfg.retry_budgets.codegen == 3
    assert cfg.retry_budgets.correctness == 3
    assert cfg.retry_budgets.performance == 3
    assert cfg.retry_budgets.interview == 1
    assert cfg.retry_budgets.polish == 1


def test_perf_target_defaults():
    cfg = SynthesisConfig()
    assert cfg.perf_target_speedup_vs_torch_compile == 1.0
    assert cfg.escalate_to_opus_on_bust is True
```

```bash
pytest tests/unit/test_config.py -v
```
Expected: ImportError.

- [ ] **Step 2: Implement `config.py`**

```python
from pydantic import BaseModel, ConfigDict, Field


class RetryBudgets(BaseModel):
    model_config = ConfigDict(frozen=True)
    interview: int = 1
    codegen: int = 3
    correctness: int = 3
    performance: int = 3
    polish: int = 1


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    retry_budgets: RetryBudgets = Field(default_factory=RetryBudgets)
    escalate_to_opus_on_bust: bool = True
    perf_target_speedup_vs_torch_compile: float = 1.0
    correctness_rtol: float = 1e-3
    correctness_atol: float = 1e-3
    nvcc_flags: tuple[str, ...] = ("-O3", "--use_fast_math")
    artifact_root: str | None = None     # default: ~/.cache/cuda_engine/runs/
    sonnet_model: str = "claude-sonnet-4-6"
    opus_model: str = "claude-opus-4-7"
    request_timeout_seconds: int = 120
```

- [ ] **Step 3: Add `SynthesisResult` to `models/reports.py`**

```python
class SynthesisResult(BaseModel):
    """Top-level return value of synthesize()."""
    passed: bool
    failed_stage: int | None = None
    failure_reason: str | None = None
    run_id: str
    artifacts_dir: str
    report: SynthesisReport
    correctness: CorrectnessReport | None = None
    performance: PerformanceReport | None = None
    kernel_callable: object | None = None    # filled in only on success

    @classmethod
    def ok(cls, *, run_id: str, artifacts_dir: str, report: SynthesisReport,
           correctness: CorrectnessReport, performance: PerformanceReport,
           kernel_callable: object) -> "SynthesisResult":
        return cls(
            passed=True, run_id=run_id, artifacts_dir=artifacts_dir, report=report,
            correctness=correctness, performance=performance, kernel_callable=kernel_callable,
        )

    @classmethod
    def failed(cls, *, stage: int, reason: str, run_id: str, artifacts_dir: str,
               report: SynthesisReport, correctness: CorrectnessReport | None = None) -> "SynthesisResult":
        return cls(
            passed=False, failed_stage=stage, failure_reason=reason,
            run_id=run_id, artifacts_dir=artifacts_dir, report=report, correctness=correctness,
        )
```

Update `models/__init__.py` to export `SynthesisResult`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/ -v
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/config.py src/cuda_engine/models/ tests/unit/test_config.py
git commit -m "feat(config): SynthesisConfig with retry budgets + SynthesisResult"
```

---

### Task 1.4: Service ABCs — `LLMClient`, `GPURunner`, `ArtifactStore`

**Files:**
- Create: `src/cuda_engine/services/__init__.py`
- Create: `src/cuda_engine/services/llm/__init__.py`
- Create: `src/cuda_engine/services/llm/base.py`
- Create: `src/cuda_engine/services/gpu/__init__.py`
- Create: `src/cuda_engine/services/gpu/base.py`
- Create: `src/cuda_engine/services/store/__init__.py`
- Create: `src/cuda_engine/services/store/base.py`
- Test: `tests/unit/services/__init__.py`, `tests/unit/services/test_abcs_exist.py`

- [ ] **Step 1: Failing test**

`tests/unit/services/test_abcs_exist.py`:
```python
import inspect
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.gpu.base import GPURunner, CompileResult, RunResult, NsightMetrics
from cuda_engine.services.store.base import ArtifactStore


def test_llm_client_is_abstract():
    assert inspect.isabstract(LLMClient)
    assert "complete" in dir(LLMClient)


def test_gpu_runner_is_abstract():
    assert inspect.isabstract(GPURunner)
    for m in ("compile", "run_kernel", "profile"):
        assert m in dir(GPURunner)


def test_store_is_abstract():
    assert inspect.isabstract(ArtifactStore)
    for m in ("new_run", "write_text", "write_bytes", "run_dir"):
        assert m in dir(ArtifactStore)
```

- [ ] **Step 2: Implement `services/llm/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[dict[str, Any]] = []
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_seconds: float = 0.0


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
```

- [ ] **Step 3: Implement `services/gpu/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class CompileResult(BaseModel):
    ok: bool
    so_path: Path | None = None
    log: str = ""
    errors: list[str] = []
    warnings: list[str] = []
    ptx_size_bytes: int = 0


class RunResult(BaseModel):
    ok: bool
    output_tensors: list[Any] | None = None      # numpy arrays
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    wall_seconds: float = 0.0


class NsightMetrics(BaseModel):
    occupancy: float | None = None
    regs_per_thread: int | None = None
    uncoalesced_global_loads_pct: float | None = None
    spill_bytes: int = 0
    achieved_bandwidth_gbps: float | None = None
    achieved_tflops: float | None = None
    raw_csv: str = ""


class GPURunner(ABC):
    @abstractmethod
    def compile(self, src: str, *, target_arch: str, extra_flags: tuple[str, ...] = ()) -> CompileResult: ...
    @abstractmethod
    def run_kernel(self, so_path: Path, inputs: list[Any], timeout_seconds: int = 30) -> RunResult: ...
    @abstractmethod
    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics: ...
```

- [ ] **Step 4: Implement `services/store/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def new_run(self) -> str: ...
    @abstractmethod
    def run_dir(self, run_id: str) -> Path: ...
    @abstractmethod
    def write_text(self, run_id: str, rel_path: str, content: str) -> Path: ...
    @abstractmethod
    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path: ...
    @abstractmethod
    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path: ...
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/services/test_abcs_exist.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cuda_engine/services tests/unit/services
git commit -m "feat(services): LLMClient, GPURunner, ArtifactStore abstract interfaces"
```

---

### Task 1.5: Mock service implementations

**Files:**
- Create: `src/cuda_engine/services/llm/mocks.py`
- Create: `src/cuda_engine/services/gpu/mocks.py`
- Create: `src/cuda_engine/services/store/mocks.py`
- Test: `tests/unit/services/llm/test_mocks.py`, `tests/unit/services/gpu/test_mocks.py`, `tests/unit/services/store/test_mocks.py`

- [ ] **Step 1: Failing tests for `MockLLMClient`**

```python
# tests/unit/services/llm/test_mocks.py
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.llm.base import LLMResponse


def test_mock_llm_returns_canned_response():
    mock = MockLLMClient(responses=["hello", "world"])
    r1 = mock.complete(system=[], messages=[], tools=None, model="claude-sonnet-4-6")
    r2 = mock.complete(system=[], messages=[], tools=None, model="claude-sonnet-4-6")
    assert r1.text == "hello"
    assert r2.text == "world"
    assert mock.call_count == 2


def test_mock_llm_supports_tool_calls():
    canned = LLMResponse(text="", tool_calls=[{"name": "compile_kernel", "input": {"src": "..."}}],
                         model="mock", tokens_in=10, tokens_out=5)
    mock = MockLLMClient(responses=[canned])
    r = mock.complete(system=[], messages=[], tools=None, model="mock")
    assert r.tool_calls[0]["name"] == "compile_kernel"
```

- [ ] **Step 2: Implement `MockLLMClient`**

```python
# src/cuda_engine/services/llm/mocks.py
from typing import Any
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[str | LLMResponse]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, *, system: list[dict[str, Any]], messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None, model: str,
        max_tokens: int = 4096, temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools, "model": model})
        if not self._responses:
            raise RuntimeError("MockLLMClient: no canned responses left")
        nxt = self._responses.pop(0)
        self.call_count += 1
        if isinstance(nxt, LLMResponse):
            return nxt
        return LLMResponse(text=nxt, model="mock", tokens_in=10, tokens_out=10)
```

- [ ] **Step 3: Failing tests for `MockGPURunner`**

```python
# tests/unit/services/gpu/test_mocks.py
from pathlib import Path
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.gpu.base import CompileResult, RunResult, NsightMetrics


def test_mock_gpu_compile_canned_results():
    canned = CompileResult(ok=True, so_path=Path("/tmp/x.so"), log="ok", ptx_size_bytes=42)
    mock = MockGPURunner(compile_results=[canned])
    r = mock.compile("kernel src", target_arch="sm_80")
    assert r.ok and r.ptx_size_bytes == 42


def test_mock_gpu_run_kernel():
    canned = RunResult(ok=True, stdout="", wall_seconds=0.001)
    mock = MockGPURunner(run_results=[canned])
    r = mock.run_kernel(Path("/tmp/x.so"), inputs=[])
    assert r.ok
```

- [ ] **Step 4: Implement `MockGPURunner`**

```python
# src/cuda_engine/services/gpu/mocks.py
from pathlib import Path
from typing import Any
from cuda_engine.services.gpu.base import GPURunner, CompileResult, RunResult, NsightMetrics


class MockGPURunner(GPURunner):
    def __init__(
        self,
        compile_results: list[CompileResult] | None = None,
        run_results: list[RunResult] | None = None,
        profile_results: list[NsightMetrics] | None = None,
    ) -> None:
        self._compile = list(compile_results or [])
        self._run = list(run_results or [])
        self._profile = list(profile_results or [])

    def compile(self, src: str, *, target_arch: str, extra_flags: tuple[str, ...] = ()) -> CompileResult:
        if not self._compile:
            return CompileResult(ok=True, so_path=Path("/tmp/mock.so"), log="ok")
        return self._compile.pop(0)

    def run_kernel(self, so_path: Path, inputs: list[Any], timeout_seconds: int = 30) -> RunResult:
        if not self._run:
            return RunResult(ok=True, wall_seconds=0.0)
        return self._run.pop(0)

    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        if not self._profile:
            return NsightMetrics(occupancy=0.5, regs_per_thread=64, raw_csv="")
        return self._profile.pop(0)
```

- [ ] **Step 5: Failing tests + `InMemoryStore`**

```python
# tests/unit/services/store/test_mocks.py
from cuda_engine.services.store.mocks import InMemoryStore


def test_in_memory_store_writes_and_reads():
    s = InMemoryStore()
    rid = s.new_run()
    p = s.write_text(rid, "stage1/prompt.md", "hello")
    assert s._files[(rid, "stage1/prompt.md")] == b"hello"
```

```python
# src/cuda_engine/services/store/mocks.py
import uuid
from pathlib import Path
from cuda_engine.services.store.base import ArtifactStore


class InMemoryStore(ArtifactStore):
    def __init__(self) -> None:
        self._files: dict[tuple[str, str], bytes] = {}

    def new_run(self) -> str:
        return uuid.uuid4().hex[:12]

    def run_dir(self, run_id: str) -> Path:
        return Path(f"<memory>/{run_id}")

    def write_text(self, run_id: str, rel_path: str, content: str) -> Path:
        self._files[(run_id, rel_path)] = content.encode()
        return self.run_dir(run_id) / rel_path

    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path:
        self._files[(run_id, rel_path)] = content
        return self.run_dir(run_id) / rel_path

    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path:
        import json
        return self.write_text(run_id, rel_path, json.dumps(obj, default=str, indent=2))
```

- [ ] **Step 6: Run all mock tests, commit**

```bash
pytest tests/unit/services -v
git add src/cuda_engine/services tests/unit/services
git commit -m "feat(services): mock implementations for LLM/GPU/Store"
```

---

### Task 1.6: Stage ABC + 5 stub stages

**Files:**
- Create: `src/cuda_engine/stages/__init__.py`, `base.py`, `interview.py`, `codegen.py`, `correctness.py`, `performance.py`, `polish.py`
- Test: `tests/unit/stages/test_stages_pass_through.py`

- [ ] **Step 1: Failing test — stub stages can be instantiated and pass-through**

```python
# tests/unit/stages/test_stages_pass_through.py
from cuda_engine.stages.interview import Stage1Interview
from cuda_engine.stages.codegen import Stage2Codegen
from cuda_engine.stages.correctness import Stage3Correctness
from cuda_engine.stages.performance import Stage4Performance
from cuda_engine.stages.polish import Stage5Polish
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.store.mocks import InMemoryStore


def test_all_stages_can_construct():
    llm = MockLLMClient(responses=[])
    gpu = MockGPURunner()
    store = InMemoryStore()
    Stage1Interview(llm=llm, store=store)
    Stage2Codegen(llm=llm, gpu=gpu, store=store)
    Stage3Correctness(llm=llm, gpu=gpu, store=store)
    Stage4Performance(llm=llm, gpu=gpu, store=store)
    Stage5Polish(llm=llm, store=store)
```

- [ ] **Step 2: Implement `stages/base.py`**

```python
from abc import ABC
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.store.base import ArtifactStore


class Stage(ABC):
    name: str = "stage"

    def __init__(
        self,
        llm: LLMClient | None = None,
        gpu: GPURunner | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        self.llm = llm
        self.gpu = gpu
        self.store = store
```

- [ ] **Step 3: Implement 5 stage stubs**

`stages/interview.py`:
```python
from cuda_engine.stages.base import Stage
from cuda_engine.models import KernelSpec, TensorArg, OptimizationPriority


class Stage1Interview(Stage):
    name = "interview"

    def run(self, *, prompt: str, reference, target_arch: str, run_id: str) -> KernelSpec:
        # M0 stub: returns a placeholder spec.
        return KernelSpec(
            name="placeholder",
            target_arch=target_arch,    # type: ignore[arg-type]
            inputs=[TensorArg(name="x", dtype="fp32", shape=("N",))],
            outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
            precision_tolerance={"rtol": 1e-3, "atol": 1e-3},
            optimization_priority=OptimizationPriority.BALANCED,
        )
```

`stages/codegen.py`:
```python
from pathlib import Path
from cuda_engine.stages.base import Stage
from cuda_engine.models import KernelSpec, KernelArtifact


class Stage2Codegen(Stage):
    name = "codegen"

    def run(self, *, spec: KernelSpec, run_id: str, retry_budget: int = 3) -> KernelArtifact:
        return KernelArtifact(kernel_cu_path=Path("/tmp/stub.cu"))
```

`stages/correctness.py`:
```python
from cuda_engine.stages.base import Stage
from cuda_engine.models import KernelSpec, KernelArtifact, CorrectnessReport


class Stage3Correctness(Stage):
    name = "correctness"

    def run(self, *, spec: KernelSpec, artifact: KernelArtifact, reference,
            run_id: str, retry_budget: int = 3) -> CorrectnessReport:
        return CorrectnessReport(passed=True, max_abs_err=0.0, max_rel_err=0.0,
                                 shapes_tested=[(128,)], failing_inputs=[])
```

`stages/performance.py`:
```python
from cuda_engine.stages.base import Stage
from cuda_engine.models import KernelSpec, KernelArtifact, PerformanceReport


class Stage4Performance(Stage):
    name = "performance"

    def run(self, *, spec: KernelSpec, artifact: KernelArtifact,
            run_id: str, retry_budget: int = 3) -> PerformanceReport:
        return PerformanceReport(speedup_vs_reference=1.0, speedup_vs_torch_compile=1.0)
```

`stages/polish.py`:
```python
from cuda_engine.stages.base import Stage
from cuda_engine.models import KernelSpec, KernelArtifact, CorrectnessReport, PerformanceReport


class Stage5Polish(Stage):
    name = "polish"

    def run(self, *, spec: KernelSpec, artifact: KernelArtifact,
            correctness: CorrectnessReport, performance: PerformanceReport,
            run_id: str) -> KernelArtifact:
        return artifact
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/unit/stages -v
git add src/cuda_engine/stages tests/unit/stages
git commit -m "feat(stages): Stage ABC + 5 stub stages (M0 pass-through)"
```

---

### Task 1.7: `Orchestrator` (M0 — happy path with mocks)

**Files:**
- Create: `src/cuda_engine/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_orchestrator.py
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.config import SynthesisConfig
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.store.mocks import InMemoryStore


def test_orchestrator_happy_path_with_mocks():
    orch = Orchestrator(
        llm=MockLLMClient(responses=[]),
        gpu=MockGPURunner(),
        store=InMemoryStore(),
        cfg=SynthesisConfig(),
    )
    result = orch.run(prompt="noop", reference=lambda x: x, target="sm_80")
    assert result.passed is True
    assert result.run_id
    assert result.report.spec_name == "placeholder"
```

- [ ] **Step 2: Implement `Orchestrator`**

```python
# src/cuda_engine/orchestrator.py
from __future__ import annotations
import time
from typing import Callable
from cuda_engine.config import SynthesisConfig
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.models import (
    SynthesisResult, SynthesisReport, CorrectnessReport, PerformanceReport,
)
from cuda_engine.stages.interview import Stage1Interview
from cuda_engine.stages.codegen import Stage2Codegen
from cuda_engine.stages.correctness import Stage3Correctness
from cuda_engine.stages.performance import Stage4Performance
from cuda_engine.stages.polish import Stage5Polish


class Orchestrator:
    def __init__(self, *, llm: LLMClient, gpu: GPURunner, store: ArtifactStore, cfg: SynthesisConfig) -> None:
        self.llm = llm
        self.gpu = gpu
        self.store = store
        self.cfg = cfg

    def run(self, *, prompt: str, reference: Callable, target: str) -> SynthesisResult:
        run_id = self.store.new_run()
        t0 = time.time()

        spec = Stage1Interview(llm=self.llm, store=self.store).run(
            prompt=prompt, reference=reference, target_arch=target, run_id=run_id,
        )

        artifact = Stage2Codegen(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec, run_id=run_id, retry_budget=self.cfg.retry_budgets.codegen,
        )

        correctness = Stage3Correctness(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec, artifact=artifact, reference=reference,
            run_id=run_id, retry_budget=self.cfg.retry_budgets.correctness,
        )
        if not correctness.passed:
            return SynthesisResult.failed(
                stage=3, reason="correctness check failed", run_id=run_id,
                artifacts_dir=str(self.store.run_dir(run_id)),
                report=SynthesisReport(run_id=run_id, spec_name=spec.name,
                                       stages_executed=["interview", "codegen", "correctness"]),
                correctness=correctness,
            )

        perf = Stage4Performance(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec, artifact=artifact, run_id=run_id,
            retry_budget=self.cfg.retry_budgets.performance,
        )

        polished = Stage5Polish(llm=self.llm, store=self.store).run(
            spec=spec, artifact=artifact, correctness=correctness, performance=perf, run_id=run_id,
        )

        report = SynthesisReport(
            run_id=run_id,
            spec_name=spec.name,
            stages_executed=["interview", "codegen", "correctness", "performance", "polish"],
            wall_time_seconds=time.time() - t0,
            warnings=["below perf target"] if perf.below_target else [],
        )
        return SynthesisResult.ok(
            run_id=run_id, artifacts_dir=str(self.store.run_dir(run_id)),
            report=report, correctness=correctness, performance=perf, kernel_callable=None,
        )
```

- [ ] **Step 3: Run test, commit**

```bash
pytest tests/unit/test_orchestrator.py -v
git add src/cuda_engine/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat(orchestrator): M0 happy-path orchestrator with mock services"
```

---

### Task 1.8: Public API — `synthesize()` + `__init__.py`

**Files:**
- Create: `src/cuda_engine/api.py`
- Modify: `src/cuda_engine/__init__.py`
- Test: `tests/unit/test_api.py`

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_api.py
from cuda_engine import synthesize, SynthesisConfig, SynthesisResult


def test_synthesize_returns_result_with_mocks(monkeypatch):
    # use the dependency-injection path so the test never hits a real backend
    from cuda_engine.services.llm.mocks import MockLLMClient
    from cuda_engine.services.gpu.mocks import MockGPURunner
    from cuda_engine.services.store.mocks import InMemoryStore

    result = synthesize(
        prompt="noop", reference=lambda x: x, target="sm_80",
        config=SynthesisConfig(),
        _llm=MockLLMClient(responses=[]),
        _gpu=MockGPURunner(),
        _store=InMemoryStore(),
    )
    assert isinstance(result, SynthesisResult)
    assert result.passed
```

- [ ] **Step 2: Implement `api.py`**

```python
# src/cuda_engine/api.py
from typing import Callable
from cuda_engine.config import SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.models import SynthesisResult


def synthesize(
    prompt: str,
    reference: Callable,
    target: str = "sm_80",
    config: SynthesisConfig | None = None,
    *,
    _llm: LLMClient | None = None,
    _gpu: GPURunner | None = None,
    _store: ArtifactStore | None = None,
) -> SynthesisResult:
    """Synthesize a CUDA kernel from English prompt + PyTorch reference.

    The `_llm`, `_gpu`, `_store` keyword arguments are injection points used by
    tests and integration runners. Real users do not pass them — defaults are
    constructed from `config`.
    """
    cfg = config or SynthesisConfig()
    if _llm is None:
        from cuda_engine.services.llm.anthropic import AnthropicClient
        _llm = AnthropicClient(cfg=cfg)
    if _gpu is None:
        from cuda_engine.services.gpu.local import LocalGPURunner
        _gpu = LocalGPURunner(cfg=cfg)
    if _store is None:
        from cuda_engine.services.store.local_dir import LocalDirStore
        _store = LocalDirStore(cfg=cfg)

    orch = Orchestrator(llm=_llm, gpu=_gpu, store=_store, cfg=cfg)
    return orch.run(prompt=prompt, reference=reference, target=target)
```

- [ ] **Step 3: Update `__init__.py`**

```python
# src/cuda_engine/__init__.py
from cuda_engine.api import synthesize
from cuda_engine.config import SynthesisConfig, RetryBudgets
from cuda_engine.models import (
    SynthesisResult, SynthesisReport, CorrectnessReport, PerformanceReport,
    KernelSpec, KernelArtifact,
)

__all__ = [
    "synthesize",
    "SynthesisConfig", "RetryBudgets",
    "SynthesisResult", "SynthesisReport", "CorrectnessReport", "PerformanceReport",
    "KernelSpec", "KernelArtifact",
]
__version__ = "0.0.1"
```

> Real `AnthropicClient` / `LocalGPURunner` / `LocalDirStore` don't exist yet — the test injects mocks. Add stub modules so import doesn't fail at module level if anyone imports `cuda_engine.services.llm.anthropic`. Step 4 covers that.

- [ ] **Step 4: Add placeholder real-impl modules (raise on call)**

Create `src/cuda_engine/services/llm/anthropic.py`:
```python
from cuda_engine.services.llm.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, cfg=None) -> None:
        raise NotImplementedError("AnthropicClient lands in M1")

    def complete(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError
```

Same shape for `services/gpu/local.py` (`LocalGPURunner`) and `services/store/local_dir.py` (`LocalDirStore`). NotImplementedError for now; M1/M2 fills them in.

- [ ] **Step 5: Run all tests + lint + types**

```bash
pytest tests/unit -v
ruff check src tests
mypy src/
```
Expected: all green / clean.

- [ ] **Step 6: Commit**

```bash
git add src/cuda_engine/api.py src/cuda_engine/__init__.py src/cuda_engine/services tests/unit/test_api.py
git commit -m "feat(api): public synthesize() entry point + module exports"
```

---

### Task 1.9: PR CI workflow

**Files:**
- Create: `.github/workflows/pr.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: PR
on: [pull_request, push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: mypy src/
      - run: pytest tests/unit -v --cov=cuda_engine --cov-report=term-missing -m "not integration"
```

- [ ] **Step 2: Commit + push to a feature branch**

```bash
git checkout -b m0/skeleton
git add .github
git commit -m "ci: PR workflow (unit + ruff + mypy)"
```

> Push to GitHub after creating the remote. Do NOT push without the user's explicit go-ahead.

---

### M0 Checkpoint — DO NOT proceed to M1 until all of these are evidence-verified

Use @superpowers:verification-before-completion. Run each command, paste output into the checkpoint reply.

- [ ] `pip install -e ".[dev]"` succeeds in a clean venv.
- [ ] `pytest tests/unit -v` reports all green and **no skipped tests**. Capture the summary line.
- [ ] `pytest tests/unit --cov=cuda_engine --cov-report=term` shows coverage ≥ 70% for `src/cuda_engine` (M0 baseline; will rise as real code lands).
- [ ] `ruff check src tests` returns 0 issues.
- [ ] `mypy src/` returns 0 issues.
- [ ] `python -c "from cuda_engine import synthesize, SynthesisConfig; print(synthesize.__doc__[:60])"` prints the docstring head.
- [ ] End-to-end mock smoke: a single Python session demonstrates `synthesize("noop", lambda x: x, "sm_80", _llm=MockLLMClient(responses=[]), _gpu=MockGPURunner(), _store=InMemoryStore()).passed is True`.
- [ ] PR CI workflow runs on a commit and reports green.

If any item fails, **fix and re-verify** — do not skip. Move to Chunk 2 only when this list is fully checked.

---

## Chunk 2 — Milestone 1: Stage 2 Real (Codegen + Compile + First Real Kernel)

**Goal:** Real `AnthropicClient` (prompt cache + tool use) and real `LocalGPURunner.compile()` wired into `Stage2Codegen`. End-to-end: vector add reference → compiled `.so` on real A100 (Colab) using real Anthropic API. Stages 1, 3, 4, 5 still stubs.

**Definition of done = M1 checkpoint from design.**

### Task 2.1: `LocalDirStore` (real)

**Files:** `src/cuda_engine/services/store/local_dir.py`, `tests/unit/services/store/test_local_dir.py`

Pattern: tmp_path fixture for filesystem tests. Round-trip text/bytes/json. Verify run dir layout from spec § 4.4. New `run_id` is short hex (12 chars).

- [ ] Write tests for: `new_run` returns 12-char hex; `write_text` creates parents and writes; `write_json` serializes pydantic + dict; `run_dir` returns absolute path.
- [ ] Implement `LocalDirStore(cfg)`: respects `cfg.artifact_root` or defaults to `~/.cache/cuda_engine/runs/`.
- [ ] Run tests; commit `feat(store): LocalDirStore filesystem-backed artifact store`.

### Task 2.2: Tool schemas

**Files:** `src/cuda_engine/services/llm/tools.py`, `tests/unit/services/llm/test_tools.py`

- [ ] Define `compile_kernel`, `run_correctness`, `nsight_profile` as `ToolSpec` constants with full JSON schemas matching the GPURunner method shapes.
- [ ] Test: every tool's `input_schema` is valid JSON-schema (use `jsonschema` lib in dev deps if needed; otherwise eyeball + Pydantic).
- [ ] Commit `feat(llm): tool schemas for compile/correctness/profile`.

### Task 2.3: `AnthropicClient` (real)

**Files:** `src/cuda_engine/services/llm/anthropic.py`, `tests/unit/services/llm/test_anthropic.py` (with `respx`/`responses`-style http mocking — pin `anthropic` SDK and mock at the transport layer; do **not** call real API in unit tests)

- [ ] Tests use a fake `anthropic.Anthropic` client (monkeypatched) returning canned responses. Cover: prompt-cache headers attached when system blocks have `cache_control`; tool definitions translated correctly to API format; tool_use response parsed into `LLMResponse.tool_calls`; token counting populated; latency captured.
- [ ] Implement `AnthropicClient`:
  - constructor: read `ANTHROPIC_API_KEY` env, create `anthropic.Anthropic` client, store `cfg`.
  - `complete()`: construct `messages.create(...)` call with system blocks (with cache_control), tools translated to Anthropic's tool format, model name, temperature/max_tokens.
  - return `LLMResponse` with text, tool_calls, model, tokens, cache_read_tokens, latency.
- [ ] Run tests (no network calls); commit.

### Task 2.4: `LocalGPURunner.compile()` (real)

**Files:** `src/cuda_engine/services/gpu/local.py`, `tests/unit/services/gpu/test_local.py`

Real `nvcc` is invoked in unit tests on the dev machine if `nvcc` is on PATH; otherwise tests are skipped with a clear marker. CI integration tests run on Colab.

- [ ] Tests:
  - Hash-cache: same `src` + same `target_arch` → second `compile()` call hits cache (verify by inspecting the runner's `cache_hits` counter).
  - Compile success path on a known-good kernel (vector add).
  - Compile failure path: deliberately broken kernel → `CompileResult.ok is False`, `errors` populated.
  - `target_arch` is translated to `-arch=sm_80` etc.
  - `extra_flags` is forwarded.
- [ ] Implement `LocalGPURunner`:
  - subprocess `nvcc`, write src to tempfile, capture stderr, parse error lines.
  - cache by `hashlib.blake2b((target_arch, flags_tuple, src).encode()).hexdigest()`.
  - cache dir: `~/.cache/cuda_engine/compile_cache/<hash>.so`.
  - run_kernel and profile raise `NotImplementedError("M2/M3")` for now.
- [ ] Commit `feat(gpu): LocalGPURunner.compile() with hash cache`.

### Task 2.5: `prompts/codegen.md`

**Files:** `src/cuda_engine/prompts/codegen.md`, `src/cuda_engine/prompts/__init__.py` (loader)

- [ ] Write the codegen system prompt: covers (a) target arch + arch knowledge for sm_80, (b) CUDA conventions (CCCL 3.x style, `cuda::buffer` where applicable, no raw `threadIdx.x` arithmetic without comment, explicit memory hierarchy choices), (c) the contract — output a single `.cu` file as a fenced code block, (d) tool-use hint: "call `compile_kernel(src)` after generating; on errors, fix and call again."
- [ ] Implement loader: `load_prompt(name) -> str` reads `prompts/<name>.md` packaged with the wheel.
- [ ] Test: loader finds `codegen.md`; non-existent name raises clearly.
- [ ] Commit `feat(prompts): codegen system prompt + loader`.

### Task 2.6: `targets/sm_80.py` knowledge file

**Files:** `src/cuda_engine/targets/sm_80.py`, `__init__.py`, `tests/unit/test_targets.py`

- [ ] Write a structured knowledge module: dict of target capabilities — supported dtypes (no FP8, no NVFP4), tensor core MMA shapes for BF16/FP16/TF32, max regs/thread, warp size 32, max threads/block 1024, shared memory per block, recommended tile sizes for elementwise (256 threads/block, 4 elements/thread), reductions (warp-shuffle then block-shared-memory two-stage).
- [ ] Test: `sm_80.CAPS["dtypes"]` includes "fp16" and excludes "fp8".
- [ ] `sm_90.py` and `sm_100.py` placeholder files: `CAPS = {"_placeholder": True}` with TODO comment.
- [ ] Commit.

### Task 2.7: `Stage2Codegen` (real)

**Files:** `src/cuda_engine/stages/codegen.py`, `tests/unit/stages/test_codegen.py`

- [ ] Tests use `MockLLMClient` returning canned tool-call sequences:
  - Happy path: LLM returns `kernel_cu` text + `compile_kernel` tool call → mocked `compile_kernel` succeeds → stage returns `KernelArtifact`.
  - 1-retry path: first compile errors → stage feeds error to LLM → 2nd attempt compiles. Verify retry count = 2.
  - Budget exhaustion: 4 compile failures → stage raises `BudgetExhaustedError` (we'll add it).
- [ ] Implement `Stage2Codegen.run`:
  - load `prompts/codegen.md` + `targets/<arch>.py` knowledge as system blocks (with cache_control).
  - user message: KernelSpec serialized + "generate kernel.cu, then call compile_kernel".
  - loop: call `llm.complete()` → if tool_calls includes `compile_kernel`, route to `gpu.compile()` → on error, append error to messages and call again; on success, return `KernelArtifact`.
  - Write each attempt to store under `stage2_codegen/attempt_NN/`.
- [ ] Commit.

### Task 2.8: Wire `Stage2Codegen` into `Orchestrator`

- [ ] Update `Orchestrator.run` to use real `Stage2Codegen` (passing through `KernelSpec` from Stage 1, which is still a stub returning a placeholder spec).
- [ ] Update `test_orchestrator` to use new mock fixtures: `MockLLMClient` with codegen-shaped responses + `MockGPURunner` with compile success.
- [ ] Commit.

### Task 2.9: Integration test — vector add on real A100

**Files:** `tests/integration/test_e2e_vector_add.py`

- [ ] Test (`@pytest.mark.integration`):
  - Real `AnthropicClient`, real `LocalGPURunner`, real `LocalDirStore`.
  - Reference: `lambda x, y: x + y`.
  - Assert: `result.passed is True` (correctness/perf are stubbed); `result.artifacts_dir / "stage2_codegen/final/kernel.so"` exists; run dir contains attempt dir(s) with `kernel.cu` + `compile.log`.
- [ ] Test only runs on machines with CUDA and `ANTHROPIC_API_KEY`. Document run command.

### Task 2.10: Run M1 checkpoint on Colab A100

This is **manual** — Colab notebook does the run, you collect evidence.

- [ ] On Colab Pro A100: clone repo, `pip install -e ".[dev]"`, set `ANTHROPIC_API_KEY`, run `pytest tests/integration/test_e2e_vector_add.py -v -m integration`.
- [ ] Inspect run dir; capture: prompt sent, kernel.cu produced, compile.log, cache_read_tokens > 0 on a 2nd run with similar prompt.
- [ ] **Induce a deliberately bad prompt** (e.g., "use fp8 fma on sm_80") to verify retry path: confirm `attempt_01` errors, `attempt_02` either fixes or surfaces with budget-exhausted (whichever; both are valid as long as the retry happened).

### M1 Checkpoint — DO NOT proceed to M2 until verified

- [ ] Vector-add e2e on Colab A100: kernel compiles, .so produced, `SynthesisResult.passed`.
- [ ] Compile cache hit verified (second identical attempt skips nvcc).
- [ ] Prompt cache hit verified (run report shows `cache_read_tokens > 0` on second run).
- [ ] Run dir layout matches spec § 4.4 (inputs/, stage2_codegen/attempt_NN/, final/).
- [ ] Compile-error retry path verified end-to-end (deliberately bad prompt produces retry, run report shows attempts > 1).
- [ ] Integration test marked `@pytest.mark.integration` is skipped by default in PR CI (verify by checking PR CI run on this branch).

---

## Chunk 3 — Milestone 2: Full Correctness Pipeline (Stages 1, 3, 5 real; hard gate)

**Goal:** Stages 1 (Interview/Opus), 3 (Correctness/HARD GATE), 5 (Polish) become real. Stage 4 stays a stub that runs a single benchmark and returns `speedup = 1.0` placeholder. Five simplest internal-regression kernels pass functional. Hard gate is enforced — induced fault returns `failed(stage=3)`.

### Task 3.1: `prompts/interview.md` + `Stage1Interview` real

- [ ] Write interview prompt: extracts shapes (call `inspect.signature(reference)` + sample-tensor introspection), dtype, target, latency-vs-throughput hint from prompt text, precision tolerance default, layout hints. Output: structured KernelSpec as JSON in fenced block.
- [ ] Use Opus 4.7 (per design). One retry budget; if Opus fails, hard error.
- [ ] Test with `MockLLMClient` returning canned spec JSON; assert parsed into `KernelSpec`; assert frozen (mutation raises).
- [ ] Reference introspection helper: `_introspect_reference(fn) -> dict` runs the fn on small dummy tensors of guessed shapes/dtypes, returns concrete shapes/dtypes. Catch and surface introspection errors as "hard structural" per design § 5.1.
- [ ] Commit.

### Task 3.2: `LocalGPURunner.run_kernel()` real (subprocess-isolated)

- [ ] Tests: load a known-good `.so` (use one compiled in M1), pass numpy inputs, assert outputs match expected; timeout test (kernel that sleeps 60s, timeout=2s, expect `timed_out=True`).
- [ ] Implement: child Python process loads `.so` via `torch.utils.cpp_extension.load`, runs forward, pickles outputs back. Hard timeout via `subprocess.run(..., timeout=...)` with `kill()` on TimeoutExpired.
- [ ] Commit.

### Task 3.3: `prompts/correctness_fix.md` + `Stage3Correctness` real (HARD GATE)

- [ ] Tests:
  - Happy path: kernel matches reference within tol → `passed=True`.
  - Mismatch path: kernel returns 2× reference → `passed=False` after retries; orchestrator returns `SynthesisResult.failed(stage=3)`.
  - Shape grid: assert at least 3 shapes tested (small/medium/large from `cfg.correctness_shape_grid`).
- [ ] Implement: generate random inputs (seeded for reproducibility) of `cfg.correctness_shape_grid` shapes, run reference (PyTorch eager) and kernel, compute `max_abs_err`/`max_rel_err`, compare to `spec.precision_tolerance`. On fail, build retry context: spec + current kernel.cu + diff stats + sample failing input → call LLM with `correctness_fix.md` system prompt + tool access to compile + run_correctness.
- [ ] Add helper `_run_pytorch_reference(fn, inputs) -> tensor` in a subprocess for safety (PyTorch + CUDA can interact unpredictably).
- [ ] Wire hard-gate semantics in `Orchestrator` (already partly there from M0; verify).
- [ ] Commit.

### Task 3.4: `prompts/polish.md` + `Stage5Polish` real

- [ ] Test: given a kernel + reports, returns annotated `kernel_annotated.cu` written to `stage5_polish/`. Annotation includes: tile-size choice, layout reasoning, perf summary, register/occupancy notes from PerformanceReport.
- [ ] Implement: simple LLM call (Sonnet, no tools), 1 retry budget. Output is annotated CUDA source.
- [ ] Commit.

### Task 3.5: First 5 internal-regression kernels (the M2 minimum set)

**Files (per kernel):** `evals/internal/<name>/{prompt.txt, reference.py, shapes.yaml, notes.md}`

Five chosen for "simplest first":
1. `vector_add_fp32` — `lambda x, y: x + y`.
2. `scalar_multiply_fp32` — `lambda x, alpha: alpha * x`.
3. `rms_norm_fp16` — RMSNorm over last dim, fp16.
4. `sum_reduction_fp32` — `lambda x: x.sum(dim=-1)`.
5. `argmax_fp32` — `lambda x: x.argmax(dim=-1)`.

For each:
- [ ] Write the kernel directory.
- [ ] Add an integration test `tests/integration/test_e2e_<name>.py` that runs `synthesize` and asserts `passed=True`.
- [ ] Run on Colab; capture results.

### Task 3.6: SynthesisReport: stage_traces wired from real stages

- [ ] Each stage updates `report.stage_traces` with model used, attempts, tokens, cache hits, latency.
- [ ] Test: orchestrator-level test asserts traces array has 5 entries on full happy path.
- [ ] Commit.

### M2 Checkpoint

- [ ] On Colab A100: 5/5 simplest kernels pass `passed=True`.
- [ ] Hard gate verified: deliberately broken `reference` (e.g., return `2*x` while prompt says "add y") → `SynthesisResult.failed(stage=3)`; run report has correctness diff + sample failing inputs.
- [ ] `KernelSpec` immutability verified: a unit test attempting to mutate frozen pydantic model raises `ValidationError`/`TypeError`.
- [ ] `report.stage_traces` populated for all 5 stages on a successful run; total tokens / cache hits / per-stage latency present.
- [ ] `Stage5Polish` produces an annotated kernel; manual eyeball verifies the annotations explain tile choice / layout.

---

## Chunk 4 — Milestone 3: Performance Loop + 30-Kernel Eval

**Goal:** Stage 4 becomes real (Nsight feedback loop), Sonnet→Opus escalation lands at the orchestrator level, full 30-kernel internal regression set runs end-to-end via `cuda-engine eval`.

### Task 4.1: `LocalGPURunner.profile()` (Nsight)

- [ ] Implement: subprocess `ncu --set basic --csv --target-processes all --kernel-name <name> ...`. Parse CSV with `csv.DictReader`. Map sections to `NsightMetrics` fields.
- [ ] Graceful degrade: if `ncu` not on PATH, return `NsightMetrics(raw_csv="ncu_not_available")` with all fields None — never raise.
- [ ] Tests: parse a captured-from-real-Nsight CSV fixture (commit a sample CSV to `tests/fixtures/`); test missing-Nsight path.
- [ ] Commit.

### Task 4.2: `prompts/perf_fix.md` + `Stage4Performance` real

- [ ] Implement benchmark protocol: warmup 5 iters, time 50 iters with `torch.cuda.Event`, take median. Run reference + kernel + `torch.compile`(reference) — three numbers.
- [ ] If median speedup < `cfg.perf_target_speedup_vs_torch_compile` (default 1.0), call `nsight_profile`, build retry context with metrics + actionable hints (templated: register pressure, occupancy, uncoalesced loads, spills).
- [ ] Up to 3 perf retries. Soft fail: ship with `below_target=True`, `warnings=[...]`.
- [ ] Tests with mocks for the loop logic; integration on Colab for the real perf path.
- [ ] Commit.

### Task 4.3: Sonnet→Opus escalation

- [ ] Update `Orchestrator`: wrap each stage call in a small helper that catches `BudgetExhaustedError` from Sonnet and re-runs the stage once with Opus.
- [ ] Logged in `report.stage_traces[i].model_used = "sonnet→opus"` and a warning.
- [ ] Tests: mock LLM that fails N+1 times on Sonnet, succeeds on Opus.
- [ ] Commit.

### Task 4.4: Fill out internal regression set to 30 kernels

Build the remaining 25 kernels in `evals/internal/`:

**Elementwise / fused (~15 more):** `layernorm_fp16`, `silu_fp16`, `gelu_fp16`, `swiglu_fp16`, `geglu_fp16`, `relu_bias_fp32`, `dropout_fp16`, `softmax_lastdim_fp16`, `rmsnorm_silu_fused_fp16`, `layernorm_silu_fused_fp16`, `bias_gelu_fp16`, `add_relu_fp32`, `sigmoid_mul_fp16`, `tanh_add_fp32`, `clamp_fp32`.

**Reductions / scans (~10 more):** `mean_lastdim_fp32`, `max_lastdim_fp32`, `min_lastdim_fp32`, `topk_fp32`, `prefix_sum_fp32`, `segment_sum_fp32`, `masked_mean_fp16`, `softmax_numerator_fp16`, `cumulative_max_fp32`, `l2_norm_fp32`.

For each: `prompt.txt`, `reference.py` (PyTorch fn), `shapes.yaml` (3+ shapes), `notes.md` (why it's here, edge cases, expected speedup band).

- [ ] Add as a single PR per ~5 kernels to keep diffs reviewable.

### Task 4.5: `evals/runner.py` + `cuda-engine eval` CLI

- [ ] Runner: discovers `evals/internal/*/`, runs `synthesize` on each, captures pass/fail + speedup numbers, writes per-kernel report + aggregate markdown + CSV to `evals/results/<date>/`.
- [ ] CLI: `cuda-engine eval --suite internal --out evals/results/2026-MM-DD/`.
- [ ] Tests: runner with 2 mock kernels; assert markdown report generated, CSV columns match, regression detection (compare to `--baseline` dir).
- [ ] Commit.

### Task 4.6: Nightly CI workflow

- [ ] `.github/workflows/nightly.yml`: cron daily, runs integration tests on a self-hosted A100 runner OR on a scheduled Colab job (document either path; default to self-hosted).
- [ ] Commit.

### M3 Checkpoint

- [ ] All 30 internal kernels run end-to-end via `cuda-engine eval --suite internal`.
- [ ] Pass rate ≥ 25/30 functional on Colab A100.
- [ ] ≥ 10/30 hit fast_1 (>1× torch.compile median).
- [ ] At least one kernel demonstrates Nsight feedback improvement (attempt-2 perf > attempt-1, in the run report).
- [ ] At least one kernel demonstrates Sonnet→Opus escalation (Opus succeeded after Sonnet bust, in stage_traces).
- [ ] Eval report at `evals/results/<date>/` contains: aggregate markdown, per-kernel JSON, CSV.

---

## Chunk 5 — Milestone 4: v1 Release Readiness

**Goal:** Full CLI, Streamlit demo, KernelBench subset, README + docs, eval gate met, ready to publish to TestPyPI.

### Task 5.1: Complete `cli.py`

- [ ] Commands: `synthesize` (single kernel from CLI args/files), `eval` (suite runner), `inspect <run_id>` (pretty-print SynthesisReport from a run dir).
- [ ] Tests with `typer.testing.CliRunner`.
- [ ] Commit.

### Task 5.2: KernelBench filter + integration

- [ ] `evals/kernelbench/filter.py`: fetches KernelBench, filters to `level1` + `level2` ops in our scope (elementwise + reductions), produces `evals/kernelbench/filtered/<name>/` directories matching the internal-regression layout.
- [ ] `evals/kernelbench/README.md`: how to fetch + filter. License notes.
- [ ] CLI: `cuda-engine eval --suite kernelbench`.
- [ ] Commit.

### Task 5.3: `examples/web_demo.py` (Streamlit)

- [ ] App: textarea for prompt, file upload for reference.py, target dropdown, "Run synthesis" button. Live status per stage. Show prompts/responses/errors as they stream. Display final kernel + report.
- [ ] Tests: smoke import + render via Streamlit's `streamlit.testing` API.
- [ ] Commit.

### Task 5.4: `examples/notebook.ipynb` (Colab quickstart)

- [ ] Cells: pip install, set API key, define a reference, call synthesize, display report. Runs end-to-end on a fresh Colab A100.
- [ ] Test by running on a fresh Colab session and capturing successful exit.

### Task 5.5: `examples/kernels/` worked examples (≥3)

- [ ] At least: `rmsnorm_silu_fp16`, `softmax_lastdim_fp16`, `topk_fp32`. Each: input prompt, generated kernel.cu (annotated), report screenshot/markdown.

### Task 5.6: Final README

- [ ] Sections: what it is (one paragraph), quickstart (`pip install` + 5-line example), eval numbers from M3/M4 runs (median speedup, fast_1 %, functional pass rate), honest scope statement (what works now, what doesn't), demo GIF, link to design doc.
- [ ] Commit.

### Task 5.7: Privacy + cost docs (`docs/privacy.md`, `docs/cost.md`)

- [ ] Privacy: prompt traces in run dirs include full LLM inputs — do not commit run dirs with proprietary references; default `~/.cache/cuda_engine/runs/` is gitignored.
- [ ] Cost: per-kernel envelope (median ~$0.10–0.30, worst case ~$2). Prompt cache implications. How to set tighter retry budgets.
- [ ] Commit.

### Task 5.8: `eval.yml` pre-release CI

- [ ] Manual-trigger workflow (`workflow_dispatch`). Runs full eval on self-hosted A100. Posts results as artifact + comment on PR.
- [ ] Commit.

### Task 5.9: Run pre-release eval; gate v1.0

- [ ] Trigger `eval.yml` on the release branch. Capture aggregate results.
- [ ] If perf bar met (per design § 6.2): tag `v1.0`, build wheel, publish to **TestPyPI** (not real PyPI yet).
- [ ] If not met: triage; fix; re-run. Do not tag until met.

### Task 5.10: TestPyPI publish + fresh-install validation

- [ ] `pip install --index-url https://test.pypi.org/simple/ cuda-engine` on a fresh Colab. Run quickstart from README. Verify end-to-end.

### M4 Checkpoint — v1.0 ships

- [ ] **Internal regression:** ≥ 95% functional, median speedup ≥ 1.0× torch.compile, p25 ≥ 0.7×, ≥ 30% fast_1.
- [ ] **KernelBench subset:** ≥ 80% functional pass rate.
- [ ] PR / nightly / pre-release CI all green.
- [ ] `pip install` from TestPyPI on a fresh Colab works end-to-end.
- [ ] Streamlit demo runs end-to-end on a fresh machine.
- [ ] README has runnable quickstart, eval numbers, honest scope statement.
- [ ] ≥ 3 worked examples in `examples/kernels/`.
- [ ] Privacy + cost docs in place.
- [ ] `evals/results/v1.0-<date>/` committed with the release artifact.

---

## Risks During Execution (carry forward from design § 9)

- Colab A100 unavailability — fallback: self-host on Lambda/Vast hourly (~$1.20/hr A100). Document in `docs/dev_setup.md`.
- Anthropic API spend during eval — pre-release eval is manual-trigger only.
- Stage 4 perf loop convergence — bounded retries; soft gate; iterate on `prompts/perf_fix.md` based on real failure cases.
- `nvcc` flag bikeshedding — pin defaults early; expose via config.

## When You Hit a Wall

Apply @superpowers:systematic-debugging:
1. Reproduce the failure with the smallest possible input.
2. Read the actual error — don't guess.
3. Check assumptions one at a time.
4. Trace the cause; don't paper over.
5. Add a regression test before fixing.
6. Fix.

Never bypass the hard gate (Stage 3 correctness). If a kernel won't verify, it doesn't ship — even if it would unblock you.

---

## Execution Order Reminder

Each chunk = one milestone = one checkpoint. Do not start chunk N+1 until checkpoint N is fully verified with evidence (real command outputs, real run dirs, real numbers). Check off the boxes literally — they're the contract between past-you and present-you.

Frequent commits per task. Each task ends with a `git commit`. The commit message convention is conventional-commits (`feat:`, `fix:`, `chore:`, `ci:`, `docs:`, `test:`).
