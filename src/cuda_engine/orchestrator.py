from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import SynthesisReport, SynthesisResult
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.stages.codegen import Stage2Codegen
from cuda_engine.stages.correctness import Stage3Correctness
from cuda_engine.stages.interview import Stage1Interview
from cuda_engine.stages.performance import Stage4Performance
from cuda_engine.stages.polish import Stage5Polish


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

    def run(self, *, prompt: str, reference: Callable[..., Any], target: str) -> SynthesisResult:
        run_id = self.store.new_run()
        started_at = time.time()

        spec = Stage1Interview(llm=self.llm, store=self.store).run(
            prompt=prompt,
            reference=reference,
            target_arch=target,
            run_id=run_id,
        )
        artifact = Stage2Codegen(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec,
            run_id=run_id,
            retry_budget=self.cfg.retry_budgets.codegen,
        )
        correctness = Stage3Correctness(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec,
            artifact=artifact,
            reference=reference,
            run_id=run_id,
            retry_budget=self.cfg.retry_budgets.correctness,
        )
        if not correctness.passed:
            return SynthesisResult.failed(
                stage=3,
                reason="correctness check failed",
                run_id=run_id,
                artifacts_dir=str(self.store.run_dir(run_id)),
                report=SynthesisReport(
                    run_id=run_id,
                    spec_name=spec.name,
                    stages_executed=["interview", "codegen", "correctness"],
                ),
                correctness=correctness,
            )

        performance = Stage4Performance(llm=self.llm, gpu=self.gpu, store=self.store).run(
            spec=spec,
            artifact=artifact,
            run_id=run_id,
            retry_budget=self.cfg.retry_budgets.performance,
        )
        Stage5Polish(llm=self.llm, store=self.store).run(
            spec=spec,
            artifact=artifact,
            correctness=correctness,
            performance=performance,
            run_id=run_id,
        )

        report = SynthesisReport(
            run_id=run_id,
            spec_name=spec.name,
            stages_executed=["interview", "codegen", "correctness", "performance", "polish"],
            wall_time_seconds=time.time() - started_at,
            warnings=["below perf target"] if performance.below_target else [],
        )
        return SynthesisResult.ok(
            run_id=run_id,
            artifacts_dir=str(self.store.run_dir(run_id)),
            report=report,
            correctness=correctness,
            performance=performance,
            kernel_callable=None,
        )
