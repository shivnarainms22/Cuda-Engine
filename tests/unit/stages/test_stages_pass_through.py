from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.codegen import Stage2Codegen
from cuda_engine.stages.correctness import Stage3Correctness
from cuda_engine.stages.interview import Stage1Interview
from cuda_engine.stages.performance import Stage4Performance
from cuda_engine.stages.polish import Stage5Polish


def test_all_stages_can_construct() -> None:
    llm = MockLLMClient(responses=[])
    gpu = MockGPURunner()
    store = InMemoryStore()

    Stage1Interview(llm=llm, store=store)
    Stage2Codegen(llm=llm, gpu=gpu, store=store)
    Stage3Correctness(llm=llm, gpu=gpu, store=store)
    Stage4Performance(llm=llm, gpu=gpu, store=store)
    Stage5Polish(llm=llm, store=store)
