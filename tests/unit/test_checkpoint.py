import pytest
from pydantic import ValidationError

from cuda_engine.checkpoint import Checkpoint, compute_fingerprint


# --- compute_fingerprint ---

def test_fingerprint_is_stable_for_same_inputs() -> None:
    fp1 = compute_fingerprint(prompt="p", reference_src="r", target="t")
    fp2 = compute_fingerprint(prompt="p", reference_src="r", target="t")
    assert fp1 == fp2


def test_fingerprint_is_hex_string_of_length_64() -> None:
    fp = compute_fingerprint(prompt="p", reference_src="r", target="t")
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_differs_when_prompt_changes() -> None:
    fp1 = compute_fingerprint(prompt="A", reference_src="r", target="t")
    fp2 = compute_fingerprint(prompt="B", reference_src="r", target="t")
    assert fp1 != fp2


def test_fingerprint_differs_when_reference_src_changes() -> None:
    fp1 = compute_fingerprint(prompt="p", reference_src="A", target="t")
    fp2 = compute_fingerprint(prompt="p", reference_src="B", target="t")
    assert fp1 != fp2


def test_fingerprint_differs_when_target_changes() -> None:
    fp1 = compute_fingerprint(prompt="p", reference_src="r", target="A")
    fp2 = compute_fingerprint(prompt="p", reference_src="r", target="B")
    assert fp1 != fp2


def test_fingerprint_parts_not_confused_by_prefix_sharing() -> None:
    # "ab" + \x00 + "c"  vs  "a" + \x00 + "bc"  must differ
    fp1 = compute_fingerprint(prompt="ab", reference_src="c", target="t")
    fp2 = compute_fingerprint(prompt="a", reference_src="bc", target="t")
    assert fp1 != fp2


# --- Checkpoint model ---

def test_checkpoint_defaults() -> None:
    cp = Checkpoint(inputs_fingerprint="abc123")
    assert cp.inputs_fingerprint == "abc123"
    assert cp.completed_stages == []
    assert cp.objects == {}


def test_checkpoint_round_trips_via_model_dump_and_validate() -> None:
    cp = Checkpoint(
        inputs_fingerprint="deadbeef",
        completed_stages=["interview", "codegen"],
        objects={"result": 42},
    )
    dumped = cp.model_dump()
    restored = Checkpoint.model_validate(dumped)
    assert restored == cp


def test_checkpoint_completed_stages_are_independent_instances() -> None:
    # default_factory must produce independent lists per instance
    cp1 = Checkpoint(inputs_fingerprint="a")
    cp2 = Checkpoint(inputs_fingerprint="b")
    cp1.completed_stages.append("x")
    assert cp2.completed_stages == []


def test_checkpoint_objects_are_independent_instances() -> None:
    cp1 = Checkpoint(inputs_fingerprint="a")
    cp2 = Checkpoint(inputs_fingerprint="b")
    cp1.objects["k"] = "v"
    assert cp2.objects == {}
