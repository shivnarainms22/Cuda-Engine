# Privacy

CUDA Engine runs Claude prompts that include your reference function source, your kernel spec, and the generated CUDA code. The full prompt/response transcripts and any artifacts produced during a run are persisted to disk so you can inspect or debug what happened. This page explains what's stored where and how to keep proprietary code from leaking.

## What gets persisted

Every `synthesize()` call creates a **run directory** under the path defined by `SynthesisConfig.artifact_root`. Default location:

```
~/.cache/cuda_engine/runs/<run_id>/
```

Inside a run directory you'll find:

```
inputs/
  prompt.txt            # the natural-language prompt
  reference.py          # source of your reference function
  config.json           # SynthesisConfig used for this run
stage1_interview/
  prompt_to_llm.md      # full system + user message sent to Sonnet
  llm_response.md       # full Sonnet response
  kernel_spec.json      # parsed KernelSpec
stage2_codegen/
  attempt_NN/
    prompt_to_llm.md    # full prompt including your reference
    llm_response.md     # full LLM response
    kernel.cu           # candidate kernel source
    compile.log         # nvcc output
  final/                # the version that compiled
stage3_correctness/...
stage4_performance/...
stage5_polish/...
report.json             # full synthesis trace
```

**The on-disk artifacts contain the entirety of what was sent to and received from Anthropic.** That includes the source of your reference function verbatim, plus the LLM's natural-language explanations of it.

## What this means in practice

- **Run directories are not gitignored at the project level by default**, but the default `artifact_root` (`~/.cache/cuda_engine/runs/`) lives outside any repo. If you change `artifact_root` to a path inside a git repo (e.g., for sharing eval outputs), add the directory to `.gitignore` if your reference code is proprietary.
- **Eval runs** (`cuda-engine eval --out <dir>`) write to whatever `--out` directory you pass. Those directories will contain your reference functions and prompts in full. Check before publishing.
- **Worked-example exports** (`examples/kernels/*/run_dir/`) intentionally include sanitized run data. If you adapt the export script for your own kernels, decide what to redact before sharing.

## Network transmission

All LLM calls go to Anthropic's API over TLS using your `ANTHROPIC_API_KEY`. Anthropic's data retention policy applies — see [Anthropic's privacy policy](https://www.anthropic.com/legal/privacy) for current terms. CUDA Engine does not transmit your code or prompts anywhere else: no telemetry, no third-party logging, no analytics.

The Anthropic SDK uses HTTPS via `urllib3`. Outbound traffic is only to `api.anthropic.com`.

## Anthropic prompt caching

System prompts and target capability blocks are sent with `cache_control: {"type": "ephemeral"}`. This means Anthropic caches them server-side for a short window (≤5 minutes) to make repeated calls cheaper. The caching does not extend Anthropic's data retention beyond their default terms — it is purely a per-account latency/cost optimization.

The `cache_read_tokens` field in `StageTrace` lets you see how many tokens were served from cache versus newly read.

## How to keep proprietary references out of run dirs

1. **Sanitize the prompt and reference** before calling `synthesize()`. Strip docstrings, comments, or naming patterns that reveal proprietary context.
2. **Use a dedicated `artifact_root`** outside any source repo when working with proprietary code, e.g. `SynthesisConfig(artifact_root="/tmp/cuda-engine-runs")`. Wipe periodically.
3. **Do not commit run directories.** The default `~/.cache/cuda_engine/runs/` is outside repos; if you redirect `artifact_root` to a repo subdirectory, ensure that subdirectory is gitignored.
4. **For eval suite runs**, use `--out` paths outside any synced or shared directory if the kernels under eval contain proprietary references.

## What CUDA Engine does NOT do

- It does not phone home, collect telemetry, or send usage data anywhere except Anthropic.
- It does not upload your kernel artifacts to any cloud service.
- It does not retain API keys outside the environment variable you set.
- It does not modify or transmit anything outside the run directory tree, the compile cache (`~/.cache/cuda_engine/compile_cache/`), and the Anthropic API.

## Reporting issues

If you find a path where proprietary content is leaving the local machine in any way other than to Anthropic, that's a bug — please file an issue.
