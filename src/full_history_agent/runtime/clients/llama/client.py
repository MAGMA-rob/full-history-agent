from typing import Any

import torch

from full_history_agent.config import ModelSettings
from ..history import get_instruction_roles
from ..loading import CausalModelClient
from ..messages import BatchedMessageCommander, get_memory_list
from .conversation import build_messages, update_clarification_memory
from .parsing import TERMINAL_TOKENS, invalid_response, parse_completion
from .tools import build_tool_descriptions


class LlamaCommander(CausalModelClient):
    def __init__(self, settings: ModelSettings, name: str = "commander") -> None:
        super().__init__(settings, name)
        self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in TERMINAL_TOKENS]
        if any(
            not isinstance(token_id, int) or token_id == self.tokenizer.unk_token_id
            for token_id in self.stop_token_ids
        ):
            raise ValueError("Llama client requires the Llama 3.1 terminal tokens")

    def _format_batch(self, message: BatchedMessageCommander) -> list[str]:
        prompts = []
        self.input_elements = []
        try:
            roles = get_instruction_roles(message)
        except Exception as error:
            for _ in message.instruction:
                prompts.append("")
                self.input_elements.append({"error": str(error)})
            return prompts
        for index, instruction in enumerate(message.instruction):
            try:
                tools = build_tool_descriptions(message.function[index], message.attributes[index])
                messages = build_messages(
                    history=message.history[index], instruction=instruction,
                    instruction_role=roles[index], attributes=message.attributes[index],
                    permanent_rules=get_memory_list(message.memory[index]), memory=message.memory[index],
                )
                elements = {"messages": messages, "tools": tools}
                prompt = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True,
                )
                prompts.append(prompt)
                self.input_elements.append(elements)
            except Exception as error:
                prompts.append("")
                self.input_elements.append({"error": str(error)})
        return prompts

    def process_batched_entry(self, message: BatchedMessageCommander, inference_mode: bool) -> list[dict[str, Any]]:
        prompts = self._format_batch(message)
        responses = [
            invalid_response(elements.get("error", "Generation did not complete"))
            for elements in self.input_elements
        ]
        raw_outputs = [""] * len(prompts)
        active = [index for index, elements in enumerate(self.input_elements) if "error" not in elements]
        if active:
            try:
                inputs = self.tokenizer(
                    [prompts[index] for index in active], return_tensors="pt", padding=True, add_special_tokens=False,
                ).to(self.input_device)
                prompt_length = inputs["input_ids"].shape[1]
                kwargs = {
                    **inputs, "max_new_tokens": self.max_new_tokens, "use_cache": self.use_cache,
                    "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.stop_token_ids,
                    "do_sample": not inference_mode,
                }
                if not inference_mode:
                    kwargs.update(temperature=0.6, top_p=0.9)
                with torch.inference_mode():
                    output = self.model.generate(**kwargs)
                if len(output) != len(active):
                    raise ValueError("Llama generation returned an unexpected batch size")
            except Exception as error:
                for index in active:
                    responses[index] = invalid_response(error)
            else:
                for output_index, index in enumerate(active):
                    try:
                        tokens = output[output_index][prompt_length:].tolist()
                        # Slice at the first terminator, retaining it and removing batch padding.
                        end = next(
                            (offset + 1 for offset, token in enumerate(tokens) if token in self.stop_token_ids),
                            len(tokens),
                        )
                        raw_outputs[index] = self.tokenizer.decode(
                            tokens[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False,
                        )
                        responses[index] = parse_completion(raw_outputs[index])
                    except Exception as error:
                        responses[index] = invalid_response(error)
        for prompt, raw, response in zip(prompts, raw_outputs, responses):
            self.log_prompt_exchange(prompt, raw, response.get("_llama_valid") is True)
            if not response.get("_llama_valid"):
                self.exchanges[-1]["error"] = response["_llama_error"]
        return responses

    def update_memory_after_response(self, memory: dict[str, Any], response: dict[str, Any]) -> None:
        update_clarification_memory(memory, response)

    def log_prompt_exchange(self, prompt: str, response: str, valid: bool) -> None:
        count = len(self.exchanges)
        try:
            super().log_prompt_exchange(prompt, response, valid)
        except Exception as error:
            if len(self.exchanges) == count:
                self.exchanges.append({"prompt": prompt, "raw_output": response, "valid": valid})
            self.exchanges[-1]["logging_error"] = str(error)

    def update_prompt_log_validity(self, validities: list[bool]) -> None:
        try:
            super().update_prompt_log_validity(validities)
        except Exception as error:
            if self.exchanges:
                self.exchanges[-1]["logging_error"] = str(error)
