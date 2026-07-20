"""Credit-free validation of the Rung 4 WMMA guidance.

Compiles kernel.cu (written by following _WMMA_GUIDANCE literally) and checks it
against torch.matmul at the exact shapes the correctness gate uses — including
the ragged, non-multiple-of-16 ones that a WMMA-only kernel fails.

Costs ZERO Anthropic credits: nvcc + torch only. A free Colab T4 (sm_75) is
enough; the 16x16x16 fp16 WMMA path is identical on sm_80.

    python tools/wmma_guidance_check/check.py

Exit 0 = the guidance we ship is compilable and numerically correct.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

# Matches SynthesisConfig.correctness_shapes (square-bound for matmul) plus the
# 4096 benchmark shape appended by Rung 3. 1, 127 and 4097 are not multiples of 16.
SHAPES = [0, 1, 127, 128, 1024, 4097, 4096]

HERE = Path(__file__).parent


def build(out: Path, *, negative_control: bool = False) -> None:
    import torch

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC",
        f"-arch={arch}", "-o", str(out), str(HERE / "kernel.cu"),
    ]
    if negative_control:
        cmd.insert(1, "-DNEGATIVE_CONTROL")  # must not land between -o and its path
    print(f"[build] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[build] FAILED — the guidance does not compile:\n")
        print(proc.stderr)
        sys.exit(1)
    print(f"[build] ok ({arch})")


def _gen(n: int, offset: int):
    """Mirrors the correctness stage's input generator: arange, remainder(17) - 8,
    scaled by 8, plus the positional arg index."""
    import torch

    v = torch.arange(n * n, dtype=torch.float32).reshape(n, n)
    return ((v.remainder(17) - 8) / 8).to(torch.float16).cuda() + offset


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("need a CUDA device (a free Colab T4 works)")
        return 2

    # --negative-control rebuilds with the col_major matrix_b defect restored. It
    # MUST fail; a pass would mean the harness cannot detect a layout bug and its
    # green result proves nothing.
    negative = "--negative-control" in sys.argv
    if negative:
        print("[mode] NEGATIVE CONTROL — this run is expected to FAIL\n")

    so = Path(tempfile.gettempdir()) / f"wmma_guidance_check{'_neg' if negative else ''}.so"
    build(so, negative_control=negative)

    lib = ctypes.CDLL(str(so))
    lib.launch_matmul_fp16.restype = ctypes.c_int
    lib.launch_matmul_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    failures = []
    for n in SHAPES:
        a, b = _gen(n, 0), _gen(n, 1)
        c = torch.zeros((n, n), dtype=torch.float16, device="cuda")

        err = lib.launch_matmul_fp16(
            ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(c.data_ptr()), n, n, n,
        )
        if err != 0:
            failures.append((n, f"cuda error {err}"))
            print(f"  N={n:<5} CUDA ERROR {err}")
            continue

        if n == 0:
            print(f"  N={n:<5} ok (no-op)")
            continue

        # Reference goes through fp16 storage too, so this measures KERNEL error
        # rather than the unavoidable fp16 output rounding (at N=4096 a result of
        # ~8000 has fp16 spacing of 8, which dwarfs any absolute tolerance).
        expected = (a.float() @ b.float()).to(torch.float16).float()
        got = c.float()
        denom = expected.abs().clamp(min=1.0)
        rel = ((got - expected).abs() / denom).max().item()
        rtol = 2e-2  # fp32 accumulate, differing summation order, fp16 store
        ok = rel <= rtol
        print(f"  N={n:<5} max_rel_err={rel:<12.5g} rtol={rtol:<8.3g} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((n, f"max_rel_err={rel:g} > rtol={rtol:g}"))

    print()
    if negative:
        # Inverted: the defect must be caught. N=1 is excluded from the verdict
        # because a 1x1 matmul is layout-invariant, so it legitimately passes.
        caught = [n for n, _ in failures if n > 1]
        if caught:
            print(f"NEGATIVE CONTROL OK — harness caught the layout defect at N={caught}.")
            print("The green run is therefore meaningful.")
            return 0
        print("NEGATIVE CONTROL FAILED — harness did NOT catch a known-bad kernel.")
        print("Its passing result proves nothing. Do not spend credits.")
        return 1

    if failures:
        print("GUIDANCE DEFECTIVE — do not spend credits yet:")
        for n, why in failures:
            print(f"  N={n}: {why}")
        return 1
    print("GUIDANCE VALID — compiles and is correct at every gate shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
