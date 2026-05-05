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
