from typing import List, Dict, Iterator, Tuple
import openai.resources
from abc import ABC, abstractmethod
import openai
from openai import OpenAI
import anthropic
from route.utils import get_registry_decorator
import time
from route.datatypes import (
    Roles,
    ChatMessage,
    ChatCompletionResponse,
    Choice,
    ChatMessageDelta,
    ChoiceDelta,
    ChatCompletionResponseChunk,
    RouterOutput,
    ModelConfig,
)
import logging
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from anthropic.lib.streaming import MessageStream
from anthropic.types.message_start_event import MessageStartEvent
from uuid import uuid4


class BaseChatHandler(ABC):

    @staticmethod
    @abstractmethod
    def _create_client(model_config: ModelConfig):
        pass

    @staticmethod
    @abstractmethod
    def _handle_system_prompt(
        messages: List[ChatMessage], model_config: ModelConfig
    ) -> List[ChatMessage]:
        pass

    @staticmethod
    @abstractmethod
    def generate(
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> ChatCompletionResponse:
        pass

    @staticmethod
    @abstractmethod
    def generate_stream(
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Iterator[ChatCompletionResponseChunk]:
        pass


CHAT_HANDLERS: Dict[str, BaseChatHandler] = {}

register = get_registry_decorator(CHAT_HANDLERS)


@register("openai")
class OpenAIChatHandler(BaseChatHandler):

    @staticmethod
    def _create_client(model_config: ModelConfig):

        api_key = model_config.get_api_key()
        base_url = model_config.get_base_url()

        if api_key or base_url:

            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
            )

        else:

            client = openai.OpenAI()

        return client

    @staticmethod
    def _handle_system_prompt(
        messages: List[ChatMessage], model_config: ModelConfig
    ) -> List[ChatMessage]:

        system_prompt = model_config.get_system_prompt()

        if system_prompt != None and messages[0].role != Roles.SYSTEM.value:

            system_message = ChatMessage(
                role=Roles.SYSTEM.value,
                content=system_prompt,
            )

            messages = [system_message] + messages

        return messages

    @staticmethod
    def _create_completion(
        client: OpenAI,
        model_config: ModelConfig,
        messages: List[ChatMessage],
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream=False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:

        completion = client.chat.completions.create(
            model=model_config.get_name(),
            messages=messages,
            temperature=model_config.get_temp() if not temp else temp,
            top_p=model_config.get_top_p() if not top_p else top_p,
            max_tokens=model_config.get_max_tokens(default=8192)
            stream=stream,
        )

        return completion

    @classmethod
    def generate(
        cls,
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> ChatCompletionResponse:

        model_config = router_output.chosen_model_config

        client = cls._create_client(model_config=model_config)

        messages = cls._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        completion: ChatCompletion = cls._create_completion(
            client=client,
            model_config=model_config,
            messages=messages,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )

        logging.info(f"{int(time.time())} Chosen Model Completion: {completion}")

        chat_completion = ChatCompletionResponse(
            id=str(completion.id),
            object="chat.completion",
            created=completion.created,
            model=completion.model,
            choices=[
                Choice(
                    index=choice.index,
                    message=ChatMessage(
                        role=choice.message.role,
                        content=choice.message.content,
                        model=router_output.chosen_model_name,
                    ),
                    finish_reason=choice.finish_reason,
                )
                for choice in completion.choices
            ],
            usage=completion.usage,
            router_outputs=router_output.model_scores,
        )

        return chat_completion

    def _skip(chunk: ChatCompletionChunk) -> bool:

        try:

            content = chunk.choices[0].delta.content

            return content == "" or content == None
        except Exception as e:
            return True

    @classmethod
    def generate_stream(
        cls,
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Iterator[ChatCompletionResponseChunk]:

        model_config = router_output.chosen_model_config

        client = cls._create_client(model_config=model_config)

        messages = cls._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        chunks: Iterator[ChatCompletionChunk] = cls._create_completion(
            client=client,
            model_config=model_config,
            messages=messages,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        first_chunk = True

        logging_content = ""

        for chunk in chunks:

            if cls._skip(chunk):
                continue

            logging_content += chunk.choices[0].delta.content

            out_chunk = ChatCompletionResponseChunk(
                id=str(chunk.id),
                object="chat.completion.chunk",
                created=chunk.created,
                model=chunk.model,
                choices=[
                    ChoiceDelta(
                        index=choice.index,
                        delta=ChatMessageDelta(
                            role=choice.delta.role,
                            content=choice.delta.content,
                            model=router_output.chosen_model_name,
                        ),
                    )
                    for choice in chunk.choices
                ],
                usage=chunk.usage,
                router_outputs=router_output.model_scores if first_chunk else None,
            ).model_dump_json()

            yield f"data: {out_chunk}\n\n"

            first_chunk = False

        logging.info(
            f"{int(time.time())} Chat Output (OpenAI Client): {logging_content}"
        )

        yield "data: [DONE]\n\n"


@register("openai-reasoning")
class OpenaiReasoningChatHandler(OpenAIChatHandler):

    @staticmethod
    def _create_completion(
        client: OpenAI,
        model_config: ModelConfig,
        messages: List[ChatMessage],
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream=False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        
        extra_field = model_config.get_extra_fields()

        # No max tokens argument
        completion = client.chat.completions.create(
            model=model_config.get_name(), messages=messages, stream=stream, reasoning_effort=extra_field.get("reasoning_effort", openai.NOT_GIVEN),
        )

        return completion


@register("openai-o1")
class OpenaiO1ChatHandler(OpenaiReasoningChatHandler):

    @classmethod
    def generate_stream(
        cls,
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Iterator[ChatCompletionResponseChunk]:

        model_config = router_output.chosen_model_config

        client = cls._create_client(model_config=model_config)

        messages = cls._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        chunk: ChatCompletion = cls._create_completion(
            client=client,
            model_config=model_config,
            messages=messages,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )

        out_chunk = ChatCompletionResponseChunk(
            id=str(chunk.id),
            object="chat.completion.chunk",
            created=chunk.created,
            model=chunk.model,
            choices=[
                ChoiceDelta(
                    index=choice.index,
                    delta=ChatMessageDelta(
                        role=choice.message.role,
                        content=choice.message.content,
                        model=router_output.chosen_model_name,
                    ),
                )
                for choice in chunk.choices
            ],
            usage=chunk.usage,
            router_outputs=router_output.model_scores,
        ).model_dump_json()

        yield f"data: {out_chunk}\n\n"

        logging.info(
            f"{int(time.time())} Chat Output (OpenAI O1 Client): {chunk.choices[0].message.content}"
        )

        yield "data: [DONE]\n\n"


@register("anthropic")
class AnthropicChatHandler(BaseChatHandler):

    @staticmethod
    def _create_client(model_config: ModelConfig):
        client = anthropic.Anthropic(api_key=model_config.get_api_key())
        return client

    @staticmethod
    @abstractmethod
    def _handle_system_prompt(
        messages: List[ChatMessage], model_config: ModelConfig
    ) -> Tuple[List[ChatMessage], str | anthropic.NotGiven]:

        system_message = model_config.get_system_prompt(default=anthropic.NOT_GIVEN)

        if system_message == None:
            system_message = anthropic.NOT_GIVEN

        if messages[0].role == Roles.SYSTEM.value:

            system_message = messages[0].content

            messages = messages[1:]

        return messages, system_message

    @staticmethod
    def generate(
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> ChatCompletionResponse:

        model_config = router_output.chosen_model_config

        client = AnthropicChatHandler._create_client(model_config=model_config)

        messages, system_message = AnthropicChatHandler._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        completion = client.messages.create(
            model=model_config.get_name(),
            messages=messages,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            temperature=model_config.get_temp() if not temp else temp,
            top_p=model_config.get_top_p() if not top_p else top_p,
            max_tokens=model_config.get_max_tokens(default=8192),
            system=system_message,
        )

        chat_completion = ChatCompletionResponse(
            id=completion.id,
            object="chat.completion",
            created=int(time.time()),
            model=completion.model,
            choices=[
                Choice(
                    index=i,
                    message=ChatMessage(
                        role=completion.role,
                        content=content.text,
                        model=router_output.chosen_model_name,
                    ),
                    finish_reason=completion.stop_reason,
                )
                for i, content in enumerate(completion.content)
            ],
            usage=completion.usage,
            router_outputs=router_output.model_scores,
        )

        return chat_completion

    @staticmethod
    def generate_stream(
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Iterator[ChatCompletionResponseChunk]:

        model_config = router_output.chosen_model_config

        client = AnthropicChatHandler._create_client(model_config=model_config)

        messages, system_message = AnthropicChatHandler._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        with client.messages.stream(
            model=model_config.get_name(),
            messages=messages,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            temperature=model_config.get_temp() if not temp else temp,
            top_p=model_config.get_top_p() if not top_p else top_p,
            max_tokens=model_config.get_max_tokens(default=8192),
            system=system_message,
        ) as _stream:

            stream: MessageStream = _stream

            # This contains the metadata
            message_start: MessageStartEvent = next(stream)

            resp_id = message_start.message.id
            model = message_start.message.model
            role = message_start.message.role

            # Ignore this useless chunk.
            next(stream)

            first_chunk = True

            logging_content = ""

            for text in stream.text_stream:

                logging_content += text

                out_chunk = ChatCompletionResponseChunk(
                    id=resp_id,
                    created=int(time.time()),
                    model=model,
                    object="chat.completion.chunk",
                    choices=[
                        ChoiceDelta(
                            delta=ChatMessageDelta(
                                content=text,
                                role=role,
                                model=router_output.chosen_model_name,
                            ),
                            index=0,
                        )
                    ],
                    router_outputs=router_output.model_scores if first_chunk else None,
                ).model_dump_json()

                yield f"data: {out_chunk}\n\n"

                first_chunk = False

            logging.info(
                f"{int(time.time())} Chat Output (Anthropic Client): {logging_content}"
            )

            yield "data: [DONE]\n\n"


import google.generativeai as genai
from google.generativeai.types.generation_types import GenerateContentResponse


@register("gemini")
class GeminiChatHandler(BaseChatHandler):

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    @staticmethod
    def _create_client(model_config: ModelConfig):

        api_key = model_config.get_api_key()

        if api_key:

            genai.configure(api_key=api_key)

    @staticmethod
    def _handle_system_prompt(
        messages: List[ChatMessage], model_config: ModelConfig
    ) -> List[ChatMessage]:

        system_prompt = model_config.get_system_prompt()

        if system_prompt != None and messages[0].role != Roles.SYSTEM.value:

            system_message = ChatMessage(
                role=Roles.SYSTEM.value,
                content=system_prompt,
            )

            messages = [system_message] + messages

        return messages

    @staticmethod
    def _create_completion(
        model_config: ModelConfig,
        messages: List[ChatMessage],
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream=False,
    ) -> GenerateContentResponse | Iterator[GenerateContentResponse]:

        generation_config = genai.GenerationConfig(
            max_output_tokens=model_config.get_max_tokens(default=8192),
            temperature=model_config.get_temp() if not temp else temp,
            top_p=model_config.get_top_p() if not top_p else top_p,
            top_k=model_config.get_top_k(),
        )

        history = []
        system_prompt = None

        for message in messages[:-1]:

            if message.role == Roles.SYSTEM.value:
                system_prompt = message.content

            elif message.role == Roles.ASSISTANT.value:
                history.append({"role": "model", "parts": message.content})

            else:
                history.append({"role": "user", "parts": message.content})

        model = genai.GenerativeModel(
            model_name=model_config.get_name(),
            system_instruction=system_prompt,
            generation_config=generation_config,
            safety_settings=GeminiChatHandler.safety_settings,
        )

        chat_session = model.start_chat(history=history)

        completion = chat_session.send_message(
            content=messages[-1].content, stream=stream
        )

        return completion

    @classmethod
    def generate(
        cls,
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> ChatCompletionResponse:

        model_config = router_output.chosen_model_config

        cls._create_client(model_config=model_config)

        messages = cls._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        completion: GenerateContentResponse = cls._create_completion(
            model_config=model_config,
            messages=messages,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )

        logging.info(f"{int(time.time())} Chosen Model Completion: {completion}")

        chat_completion = ChatCompletionResponse(
            id=str(uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=model_config.get_name(),
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role=Roles.ASSISTANT.value,
                        content=completion.text,
                        model=router_output.chosen_model_name,
                    ),
                    finish_reason="STOP",
                )
            ],
            router_outputs=router_output.model_scores,
        )

        return chat_completion

    @classmethod
    def generate_stream(
        cls,
        messages: List[ChatMessage],
        router_output: RouterOutput,
        temp: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Iterator[ChatCompletionResponseChunk]:

        model_config = router_output.chosen_model_config

        cls._create_client(model_config=model_config)

        messages = cls._handle_system_prompt(
            messages=messages, model_config=model_config
        )

        chunks: Iterator[GenerateContentResponse] = cls._create_completion(
            model_config=model_config,
            messages=messages,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        first_chunk = True

        chat_id = str(uuid4())

        logging_content = ""

        for chunk in chunks:

            logging_content += chunk.text

            out_chunk = ChatCompletionResponseChunk(
                id=chat_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_config.get_name(),
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=ChatMessageDelta(
                            role=Roles.ASSISTANT.value,
                            content=chunk.text,
                            model=router_output.chosen_model_name,
                        ),
                    )
                ],
                router_outputs=router_output.model_scores if first_chunk else None,
            ).model_dump_json()

            yield f"data: {out_chunk}\n\n"

            first_chunk = False

        logging.info(
            f"{int(time.time())} Chat Output (Gemini Client): {logging_content}"
        )

        yield "data: [DONE]\n\n"
