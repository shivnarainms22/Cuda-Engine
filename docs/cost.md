# Cost

CUDA Engine drives 4–7 Anthropic API calls per kernel under default settings, plus subprocess-isolated GPU work. This page gives you a realistic cost envelope, explains where the spend goes, and shows how to tighten retry budgets when you want a cheaper run.

## Per-kernel envelope (default config)

| Scenario | Cost (USD) | What happened |
|---|---|---|
| **Happy path** | ~$0.10–0.20 | Sonnet handles all 5 stages on the first try, no retries |
| **Typical** | ~$0.15–0.40 | 1-2 codegen retries (compile errors), 0-1 perf-fix iterations |
| **Hard kernel** | ~$0.30–0.80 | 3 codegen retries, 3 perf-fix iterations, no Opus escalation |
| **Opus escalation** | ~$0.80–2.00 | Sonnet exhausts budget on codegen OR Stage 4 perf loop ends below target, Opus retries with fresh budget |
| **Worst case** | ~$2.50 | Codegen busts to Opus, Opus also iterates, Stage 4 burns full budget on both models |

These numbers assume Claude Sonnet 4.6 at current API pricing (~$3/MTok input, ~$15/MTok output) and Claude Opus 4.7 (~$15/MTok input, ~$75/MTok output). Check [Anthropic pricing](https://www.anthropic.com/pricing) for current rates.

## Where the spend goes

For a typical happy-path run (~$0.15):

| Stage | Calls | Input tokens | Output tokens | ~$ |
|---|---|---|---|---|
| Stage 1 Interview | 1 | ~1500 | ~600 | $0.014 |
| Stage 2 Codegen | 1-3 | ~3000 each | ~1200 each | $0.045 |
| Stage 3 Correctness | 0 | — | — | $0 |
| Stage 4 Performance | 0-3 | ~5000 each | ~1500 each | $0.067 |
| Stage 5 Polish | 1 | ~3000 | ~1500 | $0.031 |

Stage 4 is the most expensive per call because it includes the current kernel source, the benchmark result, and Nsight metrics in every retry's prompt.

## Prompt caching

The system prompt and target capability blocks are sent with `cache_control: {"type": "ephemeral"}`. Anthropic caches these for ≤5 minutes per account, so the second and subsequent calls within the same run see significant savings on input tokens.

You can see cache hits in the run report:

```python
report = synthesize(...).report
for trace in report.stage_traces:
    print(trace.stage_name, trace.tokens_in, "→", trace.cache_read_tokens, "from cache")
```

Typical cache benefit on a happy-path run: ~30-50% off input token cost from Stage 2 onward.

## Eval suite cost (30 kernels)

| Scenario | Total cost | Notes |
|---|---|---|
| Happy run, all 30 pass quickly | ~$5–8 | Most M3 eval runs land here |
| Typical eval | ~$10–20 | Some retries, occasional Opus escalation |
| Worst case | ~$50–80 | Most kernels burn full budgets + Opus escalation |

The internal eval suite runner (`cuda-engine eval --suite internal`) does not pre-warm any cache across kernels — each kernel is a fresh synthesize() call. To minimize cost, run sequentially rather than parallel.

## How to reduce cost

### 1. Tighten retry budgets

The biggest knob. Default `SynthesisConfig.retry_budgets`:

```python
class RetryBudgets:
    interview: int = 1
    codegen: int = 3       # ← biggest cost lever
    correctness: int = 3   # ← drives codegen-repair loops
    performance: int = 3   # ← drives perf-fix iterations
    polish: int = 1
```

For exploratory work where you'd rather see a failure quickly than pay for retries:

```python
cfg = SynthesisConfig(
    retry_budgets=RetryBudgets(codegen=1, correctness=1, performance=1),
    opus_retry_budget_codegen=0,
    opus_retry_budget_performance=0,
)
```

This caps a run at ~$0.10. Trade-off: kernels that would have succeeded after a retry will now fail.

### 2. Disable Opus escalation

```python
cfg = SynthesisConfig(escalate_to_opus_on_bust=False)
```

Removes the Opus fallback for codegen busts and Stage 4 below-target. Bounds worst-case cost. Trade-off: lower success rate on hard kernels.

### 3. Skip Stage 4 entirely

If you only need a correct kernel (not a fast one):

```python
cfg = SynthesisConfig(retry_budgets=RetryBudgets(performance=0))
```

Stage 4 still runs the initial benchmark but doesn't iterate. Saves the most-expensive stage entirely.

### 4. Use Sonnet only

The cli/orchestrator currently lets you pass alternative model names via config. If you want to force Sonnet-only with no escalation, pair `escalate_to_opus_on_bust=False` with the default `sonnet_model="claude-sonnet-4-6"`.

## Budget guardrails

CUDA Engine does not enforce a hard dollar limit. The Anthropic API will return `BadRequestError: credit balance is too low` if your account runs out of credits mid-run — this is captured cleanly as a failure in the eval report (`failure_kind=external_error`).

For long unattended runs (nightly CI, batch evals), set your Anthropic account's spending limit in the web console.

## Observability

Every `SynthesisResult` includes:
- `report.total_llm_tokens_in` / `total_llm_tokens_out` — aggregate token counts
- `report.stage_traces[*].tokens_in/out` — per-stage breakdown
- `report.stage_traces[*].cache_read_tokens` — savings from prompt caching

You can compute spend directly from these:

```python
input_cost = (report.total_llm_tokens_in / 1e6) * sonnet_input_rate
output_cost = (report.total_llm_tokens_out / 1e6) * sonnet_output_rate
```

The eval CSV `results.csv` does not include cost columns directly today. If that's important to you, compute it from the per-kernel `report.json` files in the run directories.
