from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from route.utils import (
    get_registry_decorator,
    query_p2l_endpoint,
    get_p2l_endpoint_models,
)
from route.datatypes import ModelConfigContainer, Roles, ChatMessage, RouterOutput
from route.cost_optimizers import COST_OPTIMIZERS, BaseCostOptimizer
import numpy as np
from scipy.special import expit


class BaseRouter(ABC):

    def __init__(
        self,
        model_config_container: ModelConfigContainer,
        cost_optimizer_type: str,
        **kwargs,
    ):
        super().__init__()
        self.model_config_container = model_config_container
        self.model_list: List[str] = None
        self.model_costs: np.ndarray[float] = None
        self.cost_optimizer: BaseCostOptimizer = COST_OPTIMIZERS[cost_optimizer_type]

    @abstractmethod
    def _get_model_scores(self, messages: List[ChatMessage]) -> np.ndarray[float]:
        pass

    def _get_previous_response_model(self, messages: List[ChatMessage]) -> str | None:

        for message in reversed(messages):

            if message.role == Roles.ASSISTANT.value:

                return message.model

        return None

    def _get_prompt(self, messages: List[ChatMessage]) -> list[str]:

        prompts = []

        for message in messages:

            if message.role == Roles.USER.value:

                prompts.append(message.content)

        if len(prompts) == 0:

            raise Exception(f"No user prompt found in messages {messages}.")

        return prompts

    def get_model_direct(self, model_name: str) -> RouterOutput:
        return RouterOutput(
            chosen_model_name=model_name,
            chosen_model_config=self.model_config_container.get_model_config(
                model_name=model_name
            ),
            model_scores=None,
        )

    def route(self, messages: List[ChatMessage], cost: float = None) -> RouterOutput:

        model_scores = self._get_model_scores(messages)

        chosen_model_name = self.cost_optimizer.select_model(
            cost, self.model_list, self.model_costs, model_scores
        )

        model_scores_dict = dict(zip(self.model_list, model_scores))

        chosen_model_config = self.model_config_container.get_model_config(
            chosen_model_name
        )

        return RouterOutput(
            chosen_model_name=chosen_model_name,
            chosen_model_config=chosen_model_config,
            model_scores=model_scores_dict,
        )


ROUTERS: Dict[str, BaseRouter] = {}

register = get_registry_decorator(ROUTERS)


@register("random")
class RandomRouter(BaseRouter):
    """For debugging and gamblers."""

    def __init__(
        self,
        model_config_container: ModelConfigContainer,
        cost_optimizer_type: str,
        **kwargs,
    ):
        super().__init__(
            model_config_container=model_config_container,
            cost_optimizer_type=cost_optimizer_type,
        )

        self.model_list = model_config_container.list_models()
        self.model_costs = np.array(model_config_container.list_costs())

    def _get_model_scores(self, messages: List[ChatMessage]) -> np.ndarray[float]:
        return np.random.uniform(0.0, 1.0, size=len(self.model_list))


@register("bt-endpoint")
class EndpointP2LRouter(BaseRouter):

    # Hardcoding this because I'm tired man...
    SAMPLING_WEIGHTS = {
        "chatgpt-4o-latest-20241120": 4,
        "o1-mini": 4,
        "o1-2024-12-17": 4,
        "gpt-4o-mini-2024-07-18": 2,
        "gemma-2-27b-it": 2,
        "gemma-2-9b-it": 2,
        "gemma-2-2b-it": 2,
        "claude-3-5-sonnet-20241022": 4,
        "claude-3-opus-20240229": 4,
        "claude-3-5-haiku-20241022": 4,
        "qwen2.5-72b-instruct": 2,
        "qwen2.5-plus-1127": 4,
        "llama-3.1-405b-instruct-bf16": 4,
        "mistral-large-2411": 4,
        "grok-2-2024-08-13": 4,
        "grok-2-mini-2024-08-13": 2,
        "deepseek-v3": 6,
        "gemini-1.5-pro-002": 4,
        "gemini-1.5-flash-002": 2,
        "gemini-1.5-flash-8b-001": 2,
        "c4ai-aya-expanse-32b": 2,
        "c4ai-aya-expanse-8b": 2,
        "athene-v2-chat": 4,
        "gemini-exp-1206": 4,
        "gemini-2.0-flash-exp": 4,
        "llama-3.3-70b-instruct": 4,
        "amazon-nova-pro-v1.0": 4,
        "amazon-nova-lite-v1.0": 2,
        "amazon-nova-micro-v1.0": 2,
        "llama-3.1-tulu-3-8b": 6,
        "llama-3.1-tulu-3-70b": 6,
        "granite-3.1-8b-instruct": 6,
        "granite-3.1-2b-instruct": 6,
    }

    def __init__(
        self,
        model_config_container: ModelConfigContainer,
        cost_optimizer_type: str,
        router_model_endpoint: str,
        router_api_key: str,
        **kwargs,
    ):
        super().__init__(
            model_config_container=model_config_container,
            cost_optimizer_type=cost_optimizer_type,
        )

        self.base_url = router_model_endpoint
        self.api_key = router_api_key

        router_model_list = get_p2l_endpoint_models(self.base_url, self.api_key)

        config_model_list = model_config_container.list_models()

        self.mask = [
            router_model in config_model_list for router_model in router_model_list
        ]

        self.q_mask = [
            router_model in self.SAMPLING_WEIGHTS for router_model in router_model_list
        ]

        self.q = np.array(
            [
                float(self.SAMPLING_WEIGHTS[router_model])
                for router_model in router_model_list
                if router_model in self.SAMPLING_WEIGHTS
            ]
        )

        self.model_list = [
            model for model, keep in zip(router_model_list, self.mask) if keep
        ]

        self.model_costs = np.array(
            [
                model_config_container.get_model_config(model).get_cost()
                for model in self.model_list
            ]
        )

    def _get_model_scores(
        self, messages: List[ChatMessage]
    ) -> Tuple[np.ndarray[float], float]:

        prompt = self._get_prompt(messages)

        p2l_output = query_p2l_endpoint(prompt, self.base_url, self.api_key)

        coefs = np.array(p2l_output["coefs"])

        return coefs

    def route(self, messages: List[ChatMessage], cost: float = None) -> RouterOutput:

        model_scores = self._get_model_scores(messages)

        router_choice_scores = model_scores[self.mask]

        router_opponent_scores = model_scores[self.q_mask]

        chosen_model_name = self.cost_optimizer.select_model(
            cost,
            self.model_list,
            self.model_costs,
            router_choice_scores,
            opponent_scores=router_opponent_scores,
            opponent_distribution=self.q,
        )

        model_scores_dict = dict(zip(self.model_list, router_choice_scores))

        chosen_model_config = self.model_config_container.get_model_config(
            chosen_model_name
        )

        return RouterOutput(
            chosen_model_name=chosen_model_name,
            chosen_model_config=chosen_model_config,
            model_scores=model_scores_dict,
        )


@register("bag-endpoint")
@register("grk-endpoint")
class EndpointP2LRouter(BaseRouter):
    def __init__(
        self,
        model_config_container: ModelConfigContainer,
        cost_optimizer_type: str,
        router_model_endpoint: str,
        router_api_key: str,
        **kwargs,
    ):
        super().__init__(
            model_config_container=model_config_container,
            cost_optimizer_type=cost_optimizer_type,
        )

        self.base_url = router_model_endpoint
        self.api_key = router_api_key

        router_model_list = get_p2l_endpoint_models(self.base_url, self.api_key)

        config_model_list = model_config_container.list_models()

        self.mask = [
            router_model in config_model_list for router_model in router_model_list
        ]

        self.model_list = [
            model for model, keep in zip(router_model_list, self.mask) if keep
        ]
        self.model_costs = np.array(
            [
                model_config_container.get_model_config(model).get_cost()
                for model in self.model_list
            ]
        )

    def _get_model_scores(self, messages: List[ChatMessage]) -> np.ndarray[float]:

        prompt = self._get_prompt(messages)

        p2l_output = query_p2l_endpoint(prompt, self.base_url, self.api_key)

        coefs = np.array(p2l_output["coefs"])

        model_scores: np.ndarray[float] = expit(coefs)

        return model_scores[self.mask]
