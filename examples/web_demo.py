"""Streamlit web demo for cuda-engine.

Run locally:
    pip install "cuda-engine[demo]"
    streamlit run examples/web_demo.py

Requires:
    - ANTHROPIC_API_KEY in the environment
    - CUDA + nvcc on the host (the UI will render without them, but
      synthesis will fail at Stage 2)

Smoke test (no UI required):
    pytest tests/unit/test_web_demo.py -v
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

# Make the cuda_engine source importable when running from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cuda_engine import SynthesisConfig, SynthesisResult, synthesize  # noqa: E402
from cuda_engine.config import RetryBudgets  # noqa: E402


def render() -> None:
    """Render the Streamlit page. Idempotent — Streamlit calls it on every rerun."""
    st.set_page_config(page_title="cuda-engine demo", page_icon="⚡", layout="wide")
    st.title("⚡ cuda-engine")
    st.caption("Natural language + a PyTorch reference → a verified CUDA kernel.")

    _render_sidebar()
    prompt_text, reference_source, target, config = _render_inputs()

    if st.button(
        "Run synthesis",
        type="primary",
        disabled=not _can_run(prompt_text, reference_source),
    ):
        _run_and_display(prompt_text, reference_source, target, config)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Environment")
        st.write(
            "Requires `ANTHROPIC_API_KEY` env var, CUDA toolchain (`nvcc`), "
            "and a working PyTorch install."
        )
        st.write(
            f"ANTHROPIC_API_KEY: {'✅ set' if os.environ.get('ANTHROPIC_API_KEY') else '❌ missing'}"
        )
        try:
            import torch  # noqa: F401

            st.write("torch: ✅ importable")
        except Exception:
            st.write("torch: ❌ import failed")
        st.write(f"nvcc: {'✅ found' if _which('nvcc') else '❌ missing'}")
        st.write(f"ncu: {'✅ found' if _which('ncu') else '❌ missing'}")


def _render_inputs() -> tuple[str, str, str, SynthesisConfig]:
    col_prompt, col_reference = st.columns(2)

    with col_prompt:
        st.markdown("### Prompt")
        prompt_text = st.text_area(
            "Describe the kernel you want",
            placeholder="Generate a fp16 RMSNorm kernel without gamma over the last dimension.",
            height=200,
        )

    with col_reference:
        st.markdown("### Reference function")
        uploaded = st.file_uploader(
            "Upload a Python file with REFERENCE or reference()",
            type=["py"],
        )
        reference_source = ""
        if uploaded is not None:
            reference_source = uploaded.getvalue().decode("utf-8")
            st.code(reference_source, language="python")
        else:
            st.caption("Or paste source below:")
            reference_source = st.text_area(
                "Reference source",
                height=140,
                placeholder=(
                    "def reference(x):\n"
                    "    return x * (x.float().pow(2).mean(-1, keepdim=True) + 1e-5)"
                    ".rsqrt().to(x.dtype)\n"
                ),
            )

    target = st.selectbox(
        "CUDA target architecture",
        options=["sm_80", "sm_90", "sm_100"],
        index=0,
        help="sm_80 = A100, sm_90 = H100, sm_100 = Blackwell B200.",
    )

    with st.expander("Advanced — retry budgets and Opus escalation"):
        codegen = st.slider("Codegen retry budget", 1, 5, 3)
        correctness = st.slider("Correctness repair budget", 0, 5, 3)
        performance = st.slider("Stage 4 perf-fix budget", 0, 5, 3)
        escalate = st.checkbox(
            "Escalate to Opus on bust",
            value=True,
            help="When Sonnet exhausts its budget, retry on Opus with the failure summary.",
        )

    config = SynthesisConfig(
        retry_budgets=RetryBudgets(
            codegen=codegen,
            correctness=correctness,
            performance=performance,
        ),
        escalate_to_opus_on_bust=escalate,
    )
    return prompt_text, reference_source, target, config


def _can_run(prompt_text: str, reference_source: str) -> bool:
    return bool(prompt_text.strip()) and bool(reference_source.strip())


def _run_and_display(
    prompt_text: str,
    reference_source: str,
    target: str,
    config: SynthesisConfig,
    *,
    synthesize_fn: Callable[..., SynthesisResult] = synthesize,
) -> None:
    try:
        reference_fn = load_reference_from_source(reference_source)
    except Exception as exc:
        st.error(f"Could not load reference function: {exc}")
        st.code(traceback.format_exc(), language="text")
        return

    status_placeholder = st.empty()
    status_placeholder.info(
        "Synthesis running — this can take several minutes for hard kernels."
    )
    started = time.monotonic()
    try:
        result = synthesize_fn(
            prompt=prompt_text,
            reference=reference_fn,
            target=target,
            config=config,
        )
    except Exception as exc:
        status_placeholder.error(f"Synthesis crashed: {exc}")
        st.code(traceback.format_exc(), language="text")
        return

    wall = time.monotonic() - started
    if result.passed:
        status_placeholder.success(
            f"✅ Synthesis passed in {wall:.1f}s — run id {result.run_id}"
        )
    else:
        status_placeholder.error(
            f"❌ Synthesis failed at stage {result.failed_stage}: {result.failure_reason}"
        )

    _display_result(result)


def _display_result(result: SynthesisResult) -> None:
    cols = st.columns(3)
    perf = result.performance
    speedup = (
        f"{perf.speedup_vs_torch_compile:.2f}x"
        if perf is not None and perf.speedup_vs_torch_compile is not None
        else "n/a"
    )
    cols[0].metric("Speedup vs torch.compile", speedup)
    cols[1].metric(
        "Correctness",
        "PASS" if result.correctness is not None and result.correctness.passed else "FAIL",
    )
    cols[2].metric("Run id", result.run_id)

    st.markdown("### Stages")
    if result.report and result.report.stage_traces:
        rows = [
            {
                "stage": trace.stage_name,
                "succeeded": trace.succeeded,
                "attempts": trace.attempts,
                "model": trace.model_used,
                "tokens_in": trace.tokens_in,
                "tokens_out": trace.tokens_out,
                "cache_read": trace.cache_read_tokens,
            }
            for trace in result.report.stage_traces
        ]
        st.dataframe(rows, use_container_width=True)

    if perf is not None and perf.notes:
        with st.expander("Performance notes"):
            for note in perf.notes:
                st.text(note)
    if perf is not None and perf.warnings:
        with st.expander("Performance warnings"):
            for warning in perf.warnings:
                st.text(warning)

    st.markdown("### Final kernel")
    kernel_path = Path(result.artifacts_dir) / "stage5_polish" / "final" / "kernel.cu"
    if not kernel_path.exists():
        kernel_path = Path(result.artifacts_dir) / "stage2_codegen" / "final" / "kernel.cu"
    if kernel_path.exists():
        st.code(kernel_path.read_text(encoding="utf-8"), language="cpp")
        st.caption(f"Source: `{kernel_path}`")
    else:
        st.warning("No final kernel.cu found; check the artifacts directory.")

    st.markdown("### Artifacts directory")
    st.code(result.artifacts_dir, language="text")


def load_reference_from_source(source: str) -> Any:
    """Exec the reference source in an isolated namespace.

    Pulls a top-level ``REFERENCE`` or ``reference`` symbol. Exposed
    (not underscore-prefixed) so unit tests can exercise it directly.
    """
    namespace: dict[str, Any] = {}
    compiled = compile(source, "<demo-reference>", "exec")
    exec(compiled, namespace)  # nosec — user-supplied reference is the contract
    reference = namespace.get("REFERENCE") or namespace.get("reference")
    if not callable(reference):
        raise ValueError("source must define REFERENCE or reference(...)")
    return reference


def _which(executable: str) -> str | None:
    import shutil

    return shutil.which(executable)


# Streamlit invokes the module top-level on each rerun.
render()
