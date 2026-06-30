from __future__ import annotations

import inspect
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from cuda_engine.checkpoint import Checkpoint, compute_fingerprint
from cuda_engine.config import SynthesisConfig
from cuda_engine.models import (
    CorrectnessReport,
    KernelArtifact,
    KernelSpec,
    PerformanceReport,
    StageTrace,
    SynthesisReport,
    SynthesisResult,
)
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.anthropic import AnthropicClient
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.llm.capabilities import ProviderCapabilities
from cuda_engine.services.llm.gemini_client import GeminiClient
from cuda_engine.services.llm.openai_client import OpenAIClient
from cuda_engine.services.llm.router import LLMRouter
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.stages.base import BudgetExhaustedError
from cuda_engine.stages.codegen import Stage2Codegen
from cuda_engine.stages.correctness import Stage3Correctness
from cuda_engine.stages.interview import Stage1Interview
from cuda_engine.stages.performance import Stage4Performance
from cuda_engine.stages.polish import Stage5Polish

T = TypeVar("T")


def build_router(cfg: SynthesisConfig) -> LLMRouter:
    """Build an LLMRouter from the current environment.

    Always registers anthropic. Registers openai only when OPENAI_API_KEY is
    present in the environment; gemini only when GEMINI_API_KEY is present.
    """
    providers: dict[str, LLMClient] = {"anthropic": AnthropicClient(cfg)}
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = OpenAIClient()
    if os.environ.get("GEMINI_API_KEY"):
        providers["gemini"] = GeminiClient()
    return LLMRouter(providers)


class Orchestrator:
    def __init__(
        self,
        *,
        llm: LLMClient,
        gpu: GPURunner,
        store: ArtifactStore,
        cfg: SynthesisConfig,
    ) -> None:
        self.llm = llm
        self.gpu = gpu
        self.store = store
        self.cfg = cfg

    def run(
        self,
        *,
        prompt: str,
        reference: Callable[..., Any],
        target: str,
        resume_run_id: str | None = None,
    ) -> SynthesisResult:
        fingerprint = compute_fingerprint(
            prompt=prompt,
            reference_src=_reference_source(reference),
            target=target,
        )

        if resume_run_id is not None:
            run_id = resume_run_id
            checkpoint, loaded_spec, loaded_artifact, loaded_correctness, loaded_performance = (
                self._load_resume_state(run_id, fingerprint=fingerprint)
            )
        else:
            run_id = self.store.new_run()
            checkpoint = Checkpoint(inputs_fingerprint=fingerprint)
            loaded_spec = None
            loaded_artifact = None
            loaded_correctness = None
            loaded_performance = None

        completed = set(checkpoint.completed_stages)
        started_at = time.time()
        llm = _TracingLLMClient(self.llm)
        stage_traces: list[StageTrace] = []
        self.store.write_text(run_id, "inputs/prompt.txt", prompt)
        self.store.write_json(run_id, "inputs/config.json", self.cfg)
        self.store.write_text(run_id, "inputs/reference.py", _reference_source(reference))

        # Fast-path: everything already completed on a previous run
        if "polish" in completed:
            assert loaded_spec is not None
            assert loaded_artifact is not None
            assert loaded_correctness is not None
            assert loaded_performance is not None
            report = _build_report(
                run_id=run_id,
                spec_name=loaded_spec.name,
                stage_traces=stage_traces,
                wall_time_seconds=time.time() - started_at,
                warnings=["below perf target"] if loaded_performance.below_target else [],
            )
            result = SynthesisResult.ok(
                run_id=run_id,
                artifacts_dir=str(self.store.run_dir(run_id)),
                report=report,
                correctness=loaded_correctness,
                performance=loaded_performance,
                kernel_callable=None,
            )
            _write_result_report(self.store, result)
            return result

        # --- Stage 1: Interview ---
        if "interview" in completed and loaded_spec is not None:
            spec = loaded_spec
        else:
            spec = _run_traced_stage(
                stage_traces,
                llm,
                "interview",
                lambda: Stage1Interview(llm=llm, store=self.store).run(
                    prompt=prompt,
                    reference=reference,
                    target_arch=target,
                    run_id=run_id,
                    model=self.cfg.stage_models.interview,
                ),
            )
            checkpoint.completed_stages.append("interview")
            checkpoint.objects["spec"] = spec.model_dump(mode="json")
            self.store.write_json(run_id, "checkpoint.json", checkpoint.model_dump(mode="json"))

        # --- Stages 2+3: Codegen + Correctness (including repair loop) ---
        if (
            "correctness" in completed
            and loaded_artifact is not None
            and loaded_correctness is not None
        ):
            artifact: KernelArtifact = loaded_artifact
            correctness: CorrectnessReport = loaded_correctness
        else:
            artifact = _run_traced_stage(
                stage_traces,
                llm,
                "codegen",
                lambda: _run_codegen_with_escalation(
                    llm=llm,
                    gpu=self.gpu,
                    store=self.store,
                    cfg=self.cfg,
                    run_args={
                        "spec": spec,
                        "run_id": run_id,
                        "retry_budget": self.cfg.retry_budgets.codegen,
                    },
                ),
            )
            correctness = _run_traced_stage(
                stage_traces,
                llm,
                "correctness",
                lambda: Stage3Correctness(llm=llm, gpu=self.gpu, store=self.store).run(
                    spec=spec,
                    artifact=artifact,
                    reference=reference,
                    run_id=run_id,
                    retry_budget=self.cfg.retry_budgets.correctness,
                    correctness_shapes=self.cfg.correctness_shapes,
                ),
                succeeded=lambda report: report.passed,
            )
            for repair_attempt in range(1, self.cfg.retry_budgets.correctness + 1):
                if correctness.passed:
                    break
                repair_dir = f"stage3_repair/attempt_{repair_attempt:02d}"
                self.store.write_json(
                    run_id,
                    f"{repair_dir}/correctness_report.json",
                    correctness.model_dump(mode="json"),
                )

                def repair_action(
                    correctness_report: CorrectnessReport = correctness,
                    repair_prefix: str = repair_dir,
                ) -> KernelArtifact:
                    return _run_codegen_with_escalation(
                        llm=llm,
                        gpu=self.gpu,
                        store=self.store,
                        cfg=self.cfg,
                        run_args={
                            "spec": spec,
                            "run_id": run_id,
                            "retry_budget": self.cfg.retry_budgets.codegen,
                            "repair_context": correctness_report,
                            "artifact_prefix": f"{repair_prefix}/codegen",
                        },
                    )

                artifact = _run_traced_stage(
                    stage_traces,
                    llm,
                    "codegen_repair",
                    repair_action,
                )

                def correctness_action(candidate: KernelArtifact = artifact) -> CorrectnessReport:
                    return Stage3Correctness(llm=llm, gpu=self.gpu, store=self.store).run(
                        spec=spec,
                        artifact=candidate,
                        reference=reference,
                        run_id=run_id,
                        retry_budget=self.cfg.retry_budgets.correctness,
                        correctness_shapes=self.cfg.correctness_shapes,
                    )

                correctness = _run_traced_stage(
                    stage_traces,
                    llm,
                    "correctness",
                    correctness_action,
                    succeeded=lambda report: report.passed,
                )

            if not correctness.passed:
                result = SynthesisResult.failed(
                    stage=3,
                    reason="correctness check failed",
                    run_id=run_id,
                    artifacts_dir=str(self.store.run_dir(run_id)),
                    report=_build_report(
                        run_id=run_id,
                        spec_name=spec.name,
                        stage_traces=stage_traces,
                        wall_time_seconds=time.time() - started_at,
                    ),
                    correctness=correctness,
                )
                _write_result_report(self.store, result)
                return result

            checkpoint.completed_stages.append("correctness")
            checkpoint.objects["artifact"] = artifact.model_dump(mode="json")
            checkpoint.objects["correctness"] = correctness.model_dump(mode="json")
            self.store.write_json(run_id, "checkpoint.json", checkpoint.model_dump(mode="json"))

        # --- Stage 4: Performance ---
        if "performance" in completed and loaded_performance is not None:
            performance: PerformanceReport = loaded_performance
            # artifact is already the perf-era artifact loaded from checkpoint
        else:
            performance, artifact = _run_traced_stage(
                stage_traces,
                llm,
                "performance",
                lambda: Stage4Performance(
                    llm=llm, gpu=self.gpu, store=self.store, cfg=self.cfg
                ).run(
                    spec=spec,
                    artifact=artifact,
                    run_id=run_id,
                    retry_budget=self.cfg.retry_budgets.performance,
                    reference=reference,
                    model=self.cfg.stage_models.performance,
                ),
            )
            checkpoint.completed_stages.append("performance")
            checkpoint.objects["performance"] = performance.model_dump(mode="json")
            checkpoint.objects["artifact"] = artifact.model_dump(mode="json")
            self.store.write_json(run_id, "checkpoint.json", checkpoint.model_dump(mode="json"))

        # --- Stage 5: Polish ---
        artifact = _run_traced_stage(
            stage_traces,
            llm,
            "polish",
            lambda: Stage5Polish(llm=llm, gpu=self.gpu, store=self.store).run(
                spec=spec,
                artifact=artifact,
                correctness=correctness,
                performance=performance,
                reference=reference,
                run_id=run_id,
                model=self.cfg.stage_models.polish,
                correctness_shapes=self.cfg.correctness_shapes,
            ),
        )
        checkpoint.completed_stages.append("polish")
        checkpoint.objects["artifact"] = artifact.model_dump(mode="json")
        self.store.write_json(run_id, "checkpoint.json", checkpoint.model_dump(mode="json"))

        report = _build_report(
            run_id=run_id,
            spec_name=spec.name,
            stage_traces=stage_traces,
            wall_time_seconds=time.time() - started_at,
            warnings=["below perf target"] if performance.below_target else [],
        )
        result = SynthesisResult.ok(
            run_id=run_id,
            artifacts_dir=str(self.store.run_dir(run_id)),
            report=report,
            correctness=correctness,
            performance=performance,
            kernel_callable=None,
        )
        _write_result_report(self.store, result)
        return result

    def _load_resume_state(
        self,
        run_id: str,
        *,
        fingerprint: str,
    ) -> tuple[
        Checkpoint,
        KernelSpec | None,
        KernelArtifact | None,
        CorrectnessReport | None,
        PerformanceReport | None,
    ]:
        """Load and validate a checkpoint for resumption.

        Raises FileNotFoundError if checkpoint.json is missing, ValueError if the
        inputs fingerprint has changed since the original run.  Applies graceful
        degrade: if a completed-stage artifact file is no longer accessible, the
        affected stage (and all later ones) are dropped from ``completed_stages``.
        """
        try:
            cp_data = self.store.read_json(run_id, "checkpoint.json")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot resume run {run_id!r}: checkpoint.json not found. "
                "The run may not exist or was interrupted before any stage completed."
            ) from None

        stored = Checkpoint.model_validate(cp_data)

        if stored.inputs_fingerprint != fingerprint:
            raise ValueError(
                f"Cannot resume run {run_id!r}: inputs have changed since the original run "
                "(fingerprint mismatch). Start a fresh run instead."
            )

        objects = stored.objects
        completed: list[str] = list(stored.completed_stages)

        spec: KernelSpec | None = None
        artifact: KernelArtifact | None = None
        correctness: CorrectnessReport | None = None
        performance: PerformanceReport | None = None

        if "interview" in completed and "spec" in objects:
            spec = KernelSpec.model_validate(objects["spec"])

        if "correctness" in completed and "artifact" in objects:
            candidate = KernelArtifact.model_validate(objects["artifact"])
            if _artifact_accessible(candidate, run_id, self.store):
                artifact = candidate
                if "correctness" in objects:
                    correctness = CorrectnessReport.model_validate(objects["correctness"])
            else:
                # Artifact file missing: demote back to post-interview only
                completed = [s for s in completed if s == "interview"]

        if "performance" in completed and "performance" in objects:
            performance = PerformanceReport.model_validate(objects["performance"])

        return (
            Checkpoint(
                inputs_fingerprint=fingerprint,
                completed_stages=completed,
                objects=objects,
            ),
            spec,
            artifact,
            correctness,
            performance,
        )


def _artifact_accessible(artifact: KernelArtifact, run_id: str, store: ArtifactStore) -> bool:
    """Return True if the artifact's .cu source is readable through the store."""
    path_key = artifact.kernel_cu_path.as_posix().replace("\\", "/")
    marker = f"<memory>/{run_id}/"
    if marker in path_key:
        rel_path = path_key.split(marker, 1)[1]
        return store.exists(run_id, rel_path)
    return artifact.kernel_cu_path.exists()


def _run_codegen_with_escalation(
    *,
    llm: _TracingLLMClient,
    gpu: GPURunner,
    store: ArtifactStore,
    cfg: SynthesisConfig,
    run_args: dict[str, Any],
) -> KernelArtifact:
    """Run Stage2Codegen with configured model, escalating to Opus on BudgetExhaustedError."""
    try:
        return Stage2Codegen(llm=llm, gpu=gpu, store=store).run(
            **run_args, model=cfg.stage_models.codegen
        )
    except BudgetExhaustedError as bust:
        if not cfg.escalate_to_opus_on_bust or cfg.opus_retry_budget_codegen <= 0:
            raise
        opus_run_args = {
            **run_args,
            "retry_budget": cfg.opus_retry_budget_codegen,
            "artifact_prefix": f"{run_args.get('artifact_prefix', 'stage2_codegen')}/escalated",
            "escalation_context": bust.summary,
        }
        return Stage2Codegen(llm=llm, gpu=gpu, store=store).run(
            **opus_run_args, model=cfg.opus_model
        )


def _reference_source(reference: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(reference)
    except OSError:
        return repr(reference)


class _TracingLLMClient(LLMClient):
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.responses: list[LLMResponse] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._inner.capabilities

    def complete(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        response = self._inner.complete(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.responses.append(response)
        return response


def _run_traced_stage(
    stage_traces: list[StageTrace],
    llm: _TracingLLMClient,
    stage_name: str,
    action: Callable[[], T],
    *,
    succeeded: Callable[[T], bool] | None = None,
) -> T:
    response_start = len(llm.responses)
    started_at = time.time()
    try:
        result = action()
    except Exception:
        responses = llm.responses[response_start:]
        stage_traces.append(_build_stage_trace(stage_name, responses, started_at, succeeded=False))
        raise

    responses = llm.responses[response_start:]
    stage_traces.append(
        _build_stage_trace(
            stage_name,
            responses,
            started_at,
            succeeded=succeeded(result) if succeeded is not None else True,
        )
    )
    return result


def _build_stage_trace(
    stage_name: str,
    responses: list[LLMResponse],
    started_at: float,
    *,
    succeeded: bool,
) -> StageTrace:
    reported_latency = sum(response.latency_seconds for response in responses)
    return StageTrace(
        stage_name=stage_name,
        attempts=max(1, len(responses)),
        succeeded=succeeded,
        model_used=_model_summary(responses),
        tokens_in=sum(response.tokens_in for response in responses),
        tokens_out=sum(response.tokens_out for response in responses),
        cache_read_tokens=sum(response.cache_read_tokens for response in responses),
        latency_seconds=reported_latency if reported_latency > 0 else time.time() - started_at,
        provider=_provider_summary(responses),
        degraded=_degraded_union(responses),
    )


def _model_summary(responses: list[LLMResponse]) -> str:
    if not responses:
        return "none"
    models: list[str] = []
    for response in responses:
        if response.model not in models:
            models.append(response.model)
    return ", ".join(models)


def _provider_summary(responses: list[LLMResponse]) -> str:
    providers: list[str] = []
    for response in responses:
        if response.provider and response.provider not in providers:
            providers.append(response.provider)
    return ", ".join(providers)


def _degraded_union(responses: list[LLMResponse]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for response in responses:
        for item in response.degraded:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _build_report(
    *,
    run_id: str,
    spec_name: str,
    stage_traces: list[StageTrace],
    wall_time_seconds: float,
    warnings: list[str] | None = None,
) -> SynthesisReport:
    return SynthesisReport(
        run_id=run_id,
        spec_name=spec_name,
        stages_executed=[trace.stage_name for trace in stage_traces],
        stage_traces=stage_traces,
        total_llm_tokens_in=sum(trace.tokens_in for trace in stage_traces),
        total_llm_tokens_out=sum(trace.tokens_out for trace in stage_traces),
        wall_time_seconds=wall_time_seconds,
        warnings=warnings or [],
    )


def _write_result_report(store: ArtifactStore, result: SynthesisResult) -> None:
    payload = result.model_dump(mode="json", exclude={"kernel_callable"})
    store.write_json(result.run_id, "report.json", payload)
