"""Classify KernelBench files by whether they fall within cuda-engine v1 scope.

cuda-engine v1 handles elementwise ops + simple fusions + reductions/scans.
It does NOT handle matmul, attention, convolutions, RNNs, or autograd.
This module walks a KernelBench checkout, inspects each Model class's
`forward` method via AST, and emits a per-file classification:

  in_scope        - elementwise/reduction op the v1 pipeline can take.
  out_of_scope    - matmul/attention/conv/etc. — outside v1.
  needs_review    - couldn't classify with confidence; human inspection
                    required before adding to evals/kernelbench/filtered/.

This is a screening tool, not an auto-converter. The output is a report
that helps you decide which KernelBench files to hand-curate into our
fixture format under `evals/kernelbench/filtered/<name>/`.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Verdict = Literal["in_scope", "out_of_scope", "needs_review"]

# Operations that disqualify a file from v1 scope.
OUT_OF_SCOPE_CALLS: frozenset[str] = frozenset(
    {
        # matmul / linear algebra
        "matmul",
        "mm",
        "bmm",
        "einsum",
        "addmm",
        "linear",
        "Linear",
        # attention
        "scaled_dot_product_attention",
        "MultiheadAttention",
        "Attention",
        # convolution
        "conv1d",
        "conv2d",
        "conv3d",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        # sequence models
        "RNN",
        "LSTM",
        "GRU",
        # other heavy ops
        "fft",
        "ifft",
        "fft2",
        "ifft2",
        "rfft",
        "irfft",
    }
)

# Operations that are typical of v1-scope kernels.
IN_SCOPE_HINTS: frozenset[str] = frozenset(
    {
        # elementwise / activations
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "softplus",
        "elu",
        "leaky_relu",
        "celu",
        "swish",
        "mish",
        # arithmetic
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "pow",
        "square",
        "clamp",
        # reductions
        "sum",
        "mean",
        "max",
        "min",
        "amax",
        "amin",
        "prod",
        "argmax",
        "argmin",
        "norm",
        "var",
        "std",
        # normalization (when not BN/LN with parameters; covered)
        "layer_norm",
        "rms_norm",
        "softmax",
        "log_softmax",
        # scan/segment
        "cumsum",
        "cumprod",
        "topk",
        "sort",
        # dropout / masking
        "dropout",
        "masked_fill",
        "masked_select",
        # type / shape (compatible with v1 pipeline)
        "to",
        "type",
        "view",
        "reshape",
        "permute",
        "transpose",
        "contiguous",
        "expand",
        "squeeze",
        "unsqueeze",
        "flatten",
    }
)


@dataclass(frozen=True)
class KernelClassification:
    """Result of scanning one KernelBench source file."""

    path: str
    relative_path: str
    verdict: Verdict
    reason: str
    out_of_scope_ops: list[str] = field(default_factory=list)
    in_scope_ops: list[str] = field(default_factory=list)
    model_class_name: str | None = None
    has_forward_method: bool = False

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def classify_kernelbench_tree(root: Path) -> list[KernelClassification]:
    """Walk a KernelBench checkout and classify every model file under levels 1+2.

    `root` is expected to be the KernelBench repo root (containing a
    `KernelBench/` directory or directly the `level1`/`level2` dirs).
    Files outside `level1/` or `level2/` are skipped (level3/4 are
    full architectures, out-of-scope for v1 regardless).
    """
    candidates = _discover_python_files(root)
    return [classify_kernelbench_file(path, root) for path in sorted(candidates)]


def _discover_python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for level_dir_name in ("level1", "level2"):
        # Search both `root/level1` and `root/KernelBench/level1` because
        # both layouts are seen in the wild.
        for parent in (root, root / "KernelBench"):
            candidate = parent / level_dir_name
            if candidate.is_dir():
                paths.extend(p for p in candidate.rglob("*.py") if p.is_file())
    # Skip the obvious non-model files.
    paths = [p for p in paths if not p.name.startswith("__")]
    return paths


def classify_kernelbench_file(path: Path, root: Path | None = None) -> KernelClassification:
    """Parse a single KernelBench Model file and return its classification."""
    relative = str(path.relative_to(root)) if root is not None else path.name
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="needs_review",
            reason=f"could not read file: {exc}",
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="needs_review",
            reason=f"syntax error: {exc}",
        )

    model_class = _find_model_class(tree)
    if model_class is None:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="needs_review",
            reason="no Model class found",
        )

    forward = _find_method(model_class, "forward")
    if forward is None:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="needs_review",
            reason=f"class {model_class.name} has no forward method",
            model_class_name=model_class.name,
        )

    out_of_scope = sorted(_collect_call_names(forward) & OUT_OF_SCOPE_CALLS)
    in_scope = sorted(_collect_call_names(forward) & IN_SCOPE_HINTS)

    if out_of_scope:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="out_of_scope",
            reason=f"forward uses out-of-scope op(s): {', '.join(out_of_scope)}",
            out_of_scope_ops=out_of_scope,
            in_scope_ops=in_scope,
            model_class_name=model_class.name,
            has_forward_method=True,
        )

    if in_scope:
        return KernelClassification(
            path=str(path),
            relative_path=relative,
            verdict="in_scope",
            reason=f"forward uses in-scope op(s): {', '.join(in_scope)}",
            in_scope_ops=in_scope,
            model_class_name=model_class.name,
            has_forward_method=True,
        )

    return KernelClassification(
        path=str(path),
        relative_path=relative,
        verdict="needs_review",
        reason="no recognized in-scope or out-of-scope ops in forward; review manually",
        model_class_name=model_class.name,
        has_forward_method=True,
    )


def _find_model_class(tree: ast.AST) -> ast.ClassDef | None:
    """Return the first class definition named 'Model' (KernelBench convention)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return node
    # Fall back to the first class definition with a forward method.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _find_method(node, "forward") is not None:
            return node
    return None


def _find_method(class_def: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in class_def.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _collect_call_names(node: ast.AST) -> set[str]:
    """Return the set of attribute/name identifiers used as call targets."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def report_to_markdown(results: list[KernelClassification]) -> str:
    """Pretty-print a markdown report grouped by verdict."""
    buckets: dict[Verdict, list[KernelClassification]] = {
        "in_scope": [],
        "out_of_scope": [],
        "needs_review": [],
    }
    for entry in results:
        buckets[entry.verdict].append(entry)

    lines = [
        "# KernelBench filter report",
        "",
        f"Total files scanned: {len(results)}",
        f"- in_scope: {len(buckets['in_scope'])}",
        f"- out_of_scope: {len(buckets['out_of_scope'])}",
        f"- needs_review: {len(buckets['needs_review'])}",
        "",
    ]
    for verdict_name in ("in_scope", "needs_review", "out_of_scope"):
        entries = buckets[verdict_name]
        if not entries:
            continue
        lines.append(f"## {verdict_name} ({len(entries)})")
        lines.append("")
        for entry in entries:
            ops_summary = ""
            if entry.in_scope_ops:
                ops_summary = f" — ops: {', '.join(entry.in_scope_ops)}"
            elif entry.out_of_scope_ops:
                ops_summary = f" — disqualifying: {', '.join(entry.out_of_scope_ops)}"
            lines.append(f"- `{entry.relative_path}` — {entry.reason}{ops_summary}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(results: list[KernelClassification], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown reports under `out_dir`. Returns (json_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "candidates.json"
    md_path = out_dir / "candidates.md"
    json_path.write_text(
        json.dumps([entry.to_json() for entry in results], indent=2),
        encoding="utf-8",
    )
    md_path.write_text(report_to_markdown(results), encoding="utf-8")
    return json_path, md_path


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernelbench-root",
        type=Path,
        required=True,
        help="Path to a local KernelBench checkout (containing level1/ and level2/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write candidates.json and candidates.md into.",
    )
    args = parser.parse_args()

    results = classify_kernelbench_tree(args.kernelbench_root)
    if not results:
        print(
            f"No KernelBench files found under {args.kernelbench_root}. "
            "Expected level1/ or level2/ subdirectories (or KernelBench/level1).",
            flush=True,
        )
        return 1
    json_path, md_path = write_report(results, args.out)
    in_scope = sum(1 for r in results if r.verdict == "in_scope")
    out_of_scope = sum(1 for r in results if r.verdict == "out_of_scope")
    review = sum(1 for r in results if r.verdict == "needs_review")
    print(
        f"Scanned {len(results)} files: "
        f"{in_scope} in_scope / {out_of_scope} out_of_scope / {review} needs_review",
        flush=True,
    )
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
