from cuda_engine.services.store.mocks import InMemoryStore


def test_in_memory_store_writes_and_reads() -> None:
    store = InMemoryStore()
    run_id = store.new_run()

    path = store.write_text(run_id, "stage1/prompt.md", "hello")

    assert path == store.run_dir(run_id) / "stage1/prompt.md"
    assert store._files[(run_id, "stage1/prompt.md")] == b"hello"
