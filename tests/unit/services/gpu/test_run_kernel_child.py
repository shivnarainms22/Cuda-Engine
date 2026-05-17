import pickle
from pathlib import Path

from cuda_engine.services.gpu import _run_kernel_child


def test_try_python_extension_forward_returns_not_found_when_module_creation_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "importlib.util.spec_from_file_location",
        lambda name, path: object(),
    )
    monkeypatch.setattr(
        "importlib.util.module_from_spec",
        lambda spec: (_ for _ in ()).throw(
            ImportError("dynamic module does not define module export function")
        ),
    )

    result = _run_kernel_child._try_python_extension_forward(Path("kernel.so"), [])

    assert result is _run_kernel_child._NOT_FOUND


def test_load_reference_from_path_returns_reference_function(tmp_path: Path) -> None:
    ref_file = tmp_path / "reference.py"
    ref_file.write_text("def reference(x):\n    return x + 1\n")

    ref = _run_kernel_child._load_reference_from_path(ref_file)

    assert callable(ref)
    assert ref(41) == 42


def test_load_reference_from_path_prefers_REFERENCE_over_reference(tmp_path: Path) -> None:
    ref_file = tmp_path / "reference.py"
    ref_file.write_text(
        "def reference(x):\n    return 'lowercase'\n"
        "def _upper(x):\n    return 'UPPER'\n"
        "REFERENCE = _upper\n"
    )

    ref = _run_kernel_child._load_reference_from_path(ref_file)

    assert ref(0) == "UPPER"


def test_load_reference_from_path_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert _run_kernel_child._load_reference_from_path(tmp_path / "missing.py") is None


def test_load_reference_from_path_returns_none_on_import_error(tmp_path: Path) -> None:
    ref_file = tmp_path / "reference.py"
    ref_file.write_text("import some_module_that_does_not_exist_12345\n")

    assert _run_kernel_child._load_reference_from_path(ref_file) is None


def test_load_reference_from_path_returns_none_when_no_reference_defined(tmp_path: Path) -> None:
    ref_file = tmp_path / "reference.py"
    ref_file.write_text("def something_else(x):\n    return x\n")

    assert _run_kernel_child._load_reference_from_path(ref_file) is None


def test_load_payload_legacy_list_format_returns_no_reference(tmp_path: Path) -> None:
    """Legacy callers wrote inputs as a bare list. Stays supported."""
    payload_path = tmp_path / "inputs.pkl"
    with payload_path.open("wb") as f:
        pickle.dump([1, 2, 3], f)

    inputs, reference = _run_kernel_child._load_payload(payload_path)

    assert inputs == [1, 2, 3]
    assert reference is None


def test_load_payload_dict_format_resolves_reference_from_path(tmp_path: Path) -> None:
    """The fix's main contract: when only reference_path is given, child loads from file."""
    ref_file = tmp_path / "reference.py"
    ref_file.write_text("def reference(x):\n    return x * 2\n")
    payload_path = tmp_path / "inputs.pkl"
    with payload_path.open("wb") as f:
        pickle.dump({"inputs": [21], "reference": None, "reference_path": str(ref_file)}, f)

    inputs, reference = _run_kernel_child._load_payload(payload_path)

    assert inputs == [21]
    assert callable(reference)
    assert reference(21) == 42


def test_load_payload_dict_format_with_direct_callable(tmp_path: Path) -> None:
    """A pickleable callable in the payload bypasses path loading.

    Uses `abs` (a builtin) so the payload is genuinely pickleable without
    needing a module-level helper; what matters is that when `reference` is
    not None, _load_payload returns it directly without ever touching
    reference_path.
    """
    payload_path = tmp_path / "inputs.pkl"
    with payload_path.open("wb") as f:
        pickle.dump(
            {
                "inputs": [-5],
                "reference": abs,
                "reference_path": str(tmp_path / "should_not_be_read.py"),
            },
            f,
        )

    inputs, reference = _run_kernel_child._load_payload(payload_path)

    assert inputs == [-5]
    assert reference is abs
    assert reference(-5) == 5


def test_load_payload_dict_with_neither_reference_nor_path(tmp_path: Path) -> None:
    """Stage 4 retry path: dict format but reference omitted entirely."""
    payload_path = tmp_path / "inputs.pkl"
    with payload_path.open("wb") as f:
        pickle.dump({"inputs": [7], "reference": None, "reference_path": None}, f)

    inputs, reference = _run_kernel_child._load_payload(payload_path)

    assert inputs == [7]
    assert reference is None
