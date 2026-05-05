from cuda_engine.targets import load_target_caps, sm_80, sm_90, sm_100


def test_sm80_capabilities_include_supported_and_excluded_dtypes() -> None:
    assert "fp16" in sm_80.CAPS["dtypes"]
    assert "fp8" not in sm_80.CAPS["dtypes"]
    assert sm_80.CAPS["warp_size"] == 32


def test_future_targets_are_placeholders() -> None:
    assert sm_90.CAPS["_placeholder"] is True
    assert sm_100.CAPS["_placeholder"] is True


def test_load_target_caps_returns_capabilities() -> None:
    assert load_target_caps("sm_80")["max_threads_per_block"] == 1024
