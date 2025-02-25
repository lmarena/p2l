from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from enum import Enum


class ModelConfig:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_name(self) -> str:
        return self.config["name"]

    def get_temp(self) -> float:
        return self.config["temp"]

    def get_top_p(self) -> float:
        return self.config["top_p"]

    def get_top_k(self, default=None) -> int:
        return self.config.get("top_k", default)

    def get_system_prompt(self, default=None) -> str | None | Any:
        return self.config.get("system_prompt", default)

    def get_api_key(self, default=None) -> str | None | Any:
        return self.config.get("api_key", default)

    def get_base_url(self, default=None) -> str | None | Any:
        return self.config.get("base_url", default)

    def get_type(self) -> str:
        return self.config["type"]

    def get_cost(self) -> float:
        return self.config["cost"]

    def get_max_tokens(self, default=None) -> int | None | Any:
        return self.config.get("max_tokens", default)
    
    def get_extra_fields(self) -> Dict:
        return self.config.get("extra_fields", {}) # Maybe should be None...

    def __repr__(self):
        return repr(
            dict(
                name=self.get_name(),
                type=self.get_type(),
                cost=self.get_cost(),
            )
        )


class ModelConfigContainer:
    def __init__(self, model_config_dicts: Dict[str, Dict[str, Any]]):
        self.model_configs: Dict[str, ModelConfig] = dict(
            (name, ModelConfig(config)) for name, config in model_config_dicts.items()
        )

    def get_model_config(self, model_name: str) -> ModelConfig:
        return self.model_configs[model_name]

    def list_models(self) -> List[str]:
        return list(self.model_configs.keys())

    def list_costs(self) -> List[float]:

        costs: List[float] = []

        for model_name in self.list_models():
            model_config = self.get_model_config(model_name)
            costs.append(model_config.get_cost())

        return costs

    def __repr__(self):
        return repr(self.model_configs)


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """
    Represents a single message in the conversation.
    role: "system", "user", or "assistant"
    content: the actual text
    """

    role: str
    content: str
    model: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """
    Request body for Chat Completion.
    """

    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    cost: Optional[float] = None
    direct_model: Optional[str] = None


class Choice(BaseModel):
    """
    Represents a single choice in the final response (non-streaming mode).
    """

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """
    Response model for non-streaming mode.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[BaseModel] = None
    router_outputs: Optional[Dict[str, float]] = None


class ChatMessageDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None


class ChoiceDelta(BaseModel):
    delta: ChatMessageDelta
    finish_reason: Optional[str] = None
    index: int


class ChatCompletionResponseChunk(BaseModel):
    id: str
    choices: List[ChoiceDelta]
    created: int
    model: str
    object: str
    usage: Optional[BaseModel] = None
    router_outputs: Optional[Dict[str, float]] = None


@dataclass
class RouterOutput:
    chosen_model_name: str
    chosen_model_config: ModelConfig
    model_scores: Dict[str, float] | None
