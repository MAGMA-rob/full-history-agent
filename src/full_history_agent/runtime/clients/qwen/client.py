from __future__ import annotations

from typing import Any, Dict, List

import torch

from full_history_agent.config import ModelSettings
from ..history import get_instruction_roles
from ..loading import CausalModelClient
from ..messages import BatchedMessageCommander, get_memory_list
from .conversation import build_messages, update_reasoning_memory
from .parsing import parse_completion
from .tools import ToolCatalog, build_tool_descriptions


class QwenCommander(CausalModelClient):
    def __init__(self, settings: ModelSettings, name: str = "commander") -> None:
        super().__init__(settings, name)
        self.enable_thinking = settings.reasoning_effort != "low"
        self.model_type = str(getattr(self.model.config, "model_type", "")).lower()

    def _format_inputs(
        self,
        message: BatchedMessageCommander,
    ) -> tuple[List[str], List[ToolCatalog], List[bool]]:
        _validate_batch(message)
        instruction_roles = get_instruction_roles(message)
        formatted_inputs = []
        catalogs = []
        new_task_flags = []
        self.input_elements = []

        for index, instruction in enumerate(message.instruction):
            tools, catalog = build_tool_descriptions(
                message.function[index],
                message.attributes[index],
            )
            messages, starts_new_task = build_messages(
                history=message.history[index],
                instruction=instruction,
                instruction_role=instruction_roles[index],
                attributes=message.attributes[index],
                permanent_rules=get_memory_list(message.memory[index]),
                memory=message.memory[index],
            )
            formatted_inputs.append(_apply_chat_template(
                self.tokenizer,
                messages,
                tools,
                self.enable_thinking,
            ))
            catalogs.append(catalog)
            new_task_flags.append(starts_new_task)
            self.input_elements.append({
                "messages": messages,
                "tools": tools,
                "enable_thinking": self.enable_thinking,
            })

        return formatted_inputs, catalogs, new_task_flags

    def _format_batch(self, message: BatchedMessageCommander) -> List[str]:
        formatted_inputs, _, _ = self._format_inputs(message)
        return formatted_inputs

    def process_batched_entry(
        self,
        message: BatchedMessageCommander,
        inference_mode: bool,
    ) -> List[Dict[str, Any]]:
        formatted_inputs, catalogs, new_task_flags = self._format_inputs(message)
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.input_device)
        prompt_length = inputs["input_ids"].shape[1]

        # Qwen explicitly advises sampling in both modes. Quality takes priority
        # over the deterministic-decoding meaning of inference_mode for this adapter.
        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": self.use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.6 if self.enable_thinking else 0.7,
            "top_p": 0.95 if self.enable_thinking else 0.8,
            "top_k": 20,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)

        responses = []
        for index, catalog in enumerate(catalogs):
            generated_tokens = output[index][prompt_length:].tolist()
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
            )
            parsed = parse_completion(
                tokenizer=self.tokenizer,
                generated_ids=generated_tokens,
                model_type=self.model_type,
                catalog=catalog,
            )
            parsed["_qwen_new_task"] = new_task_flags[index]
            valid = parsed.get("_qwen_valid") is True
            self.log_prompt_exchange(
                formatted_inputs[index],
                response_text,
                valid,
            )
            responses.append(parsed)
        return responses

    def update_memory_after_response(
        self,
        memory: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        update_reasoning_memory(memory, response)


def _validate_batch(message: BatchedMessageCommander) -> None:
    batch_size = len(message.instruction)
    if not batch_size:
        raise ValueError("BatchedMessageCommander must contain at least one instruction")
    for field_name in ("memory", "attributes", "history", "function"):
        field_value = getattr(message, field_name)
        if len(field_value) != batch_size:
            raise ValueError(
                f"{field_name} must have the same length as instruction "
                f"({len(field_value)} != {batch_size})"
            )


def _apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)
