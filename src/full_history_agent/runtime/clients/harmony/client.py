from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from openai_harmony import (
    HarmonyEncodingName,
    ReasoningEffort,
    Role,
    load_harmony_encoding,
)

from full_history_agent.config import ModelSettings
from ..loading import CausalModelClient
from ..messages import BatchedMessageCommander, get_memory_list
from ..history import get_instruction_roles
from .conversation import build_conversation, history_window_start, update_analysis_memory
from .parsing import parse_completion
from .tools import ToolCatalog, build_tool_descriptions


REASONING_EFFORTS = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


class HarmonyCommander(CausalModelClient):
    def __init__(self, settings: ModelSettings, name: str = "commander") -> None:
        self.encoding = self._load_harmony_encoding()
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.history_max_messages = settings.history_max_messages
        self.reasoning_effort = REASONING_EFFORTS[settings.reasoning_effort]
        super().__init__(settings, name)

    def _format_tokens(
        self,
        message: BatchedMessageCommander,
    ) -> tuple[List[List[int]], List[ToolCatalog]]:
        self._validate_batch(message)
        instruction_roles = get_instruction_roles(message)
        prefill_ids: List[List[int]] = []
        catalogs: List[ToolCatalog] = []
        self.input_elements = []

        for index, instruction in enumerate(message.instruction):
            tools, catalog = build_tool_descriptions(
                message.function[index],
                message.attributes[index],
            )
            memory = message.memory[index]
            conversation = build_conversation(
                history=message.history[index],
                instruction=instruction,
                instruction_role=instruction_roles[index],
                attributes=message.attributes[index],
                permanent_rules=get_memory_list(memory),
                tools=tools,
                reasoning_effort=self.reasoning_effort,
                memory=memory,
                history_max_messages=self.history_max_messages,
            )
            prefill_ids.append(
                self.encoding.render_conversation_for_completion(
                    conversation,
                    Role.ASSISTANT,
                )
            )
            catalogs.append(catalog)
            start_index = history_window_start(message.history[index], self.history_max_messages)
            self.input_elements.append({
                "history": message.history[index][start_index:],
                "history_start_index": start_index,
                "instruction": instruction,
                "instruction_role": instruction_roles[index],
                "attributes": message.attributes[index],
                "permanent_rules": get_memory_list(memory),
                "tools": [tool.model_dump(mode="json") for tool in tools],
            })

        return prefill_ids, catalogs

    def _format_batch(self, message: BatchedMessageCommander) -> List[str]:
        prefill_ids, _ = self._format_tokens(message)
        return [self.encoding.decode(tokens) for tokens in prefill_ids]

    def process_batched_entry(
        self,
        message: BatchedMessageCommander,
        inference_mode: bool,
    ) -> List[Dict[str, Any]]:
        prefill_ids, catalogs = self._format_tokens(message)
        inputs = self._pad_prefill_ids(prefill_ids)

        with torch.inference_mode():
            output = self.model.generate(
                **self._generation_kwargs(inputs, inference_mode),
            )

        prompt_length = inputs["input_ids"].shape[1]
        responses = []
        for index, catalog in enumerate(catalogs):
            completion_ids = output[index][prompt_length:].tolist()
            parsed = parse_completion(
                self.encoding,
                completion_ids,
                self._completion_terminal_token_ids(),
                catalog,
            )
            valid = (
                parsed.get("_gpt_oss_valid") is True
                and isinstance(parsed.get("action", {}), (dict, list))
            )
            self.log_prompt_exchange(
                self.encoding.decode(prefill_ids[index]),
                self.encoding.decode(completion_ids),
                valid,
            )
            responses.append(parsed)
        return responses

    def update_memory_after_response(
        self,
        memory: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        update_analysis_memory(memory, response)

    def _pad_prefill_ids(
        self,
        batched_ids: Sequence[Sequence[int]],
    ) -> Dict[str, torch.Tensor]:
        max_length = max(len(ids) for ids in batched_ids)
        pad_token_id = self._pad_token_id()
        input_ids = []
        attention_mask = []
        for ids in batched_ids:
            pad_length = max_length - len(ids)
            input_ids.append([pad_token_id] * pad_length + list(ids))
            attention_mask.append([0] * pad_length + [1] * len(ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.input_device),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
                device=self.input_device,
            ),
        }

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            return int(eos_token_id[0])
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    def _generation_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        inference_mode: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": self.use_cache,
            "pad_token_id": self._pad_token_id(),
        }
        stop_token_ids = self._generation_stop_token_ids()
        if stop_token_ids:
            kwargs["eos_token_id"] = stop_token_ids
        if inference_mode:
            kwargs["do_sample"] = False
        else:
            kwargs.update({
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
            })
        return kwargs

    def _generation_stop_token_ids(self) -> List[int]:
        return unique_token_ids(
            list(self.stop_token_ids)
            + token_id_list(getattr(self.tokenizer, "eos_token_id", None))
        )

    def _completion_terminal_token_ids(self) -> List[int]:
        return unique_token_ids(
            self._generation_stop_token_ids()
            + token_id_list(getattr(self.tokenizer, "pad_token_id", None))
        )

    @staticmethod
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

    @staticmethod
    def _load_harmony_encoding() -> Any:
        try:
            return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        except Exception as error:
            raise RuntimeError(
                "HarmonyCommander could not initialize the GPT-OSS Harmony encoding. "
                "Make sure openai-harmony can access or cache its vocabulary."
            ) from error


def token_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        token_ids = []
        for item in value:
            token_ids.extend(token_id_list(item))
        return token_ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def unique_token_ids(token_ids: Sequence[int]) -> List[int]:
    return list(dict.fromkeys(token_ids))
