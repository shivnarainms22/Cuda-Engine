from pydantic import BaseModel, ConfigDict


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    prompt_caching: bool = False
    tool_use: bool = False
    max_context: int = 200_000
