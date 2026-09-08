from full_history_agent.config import ModelSettings
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import json, re
from pathlib import Path

from .messages import BatchedMessageCommander, get_memory_list
from ..loading import CausalModelClient
from .history import format_history_content, get_instruction_roles, map_chat_role

from transformers import BitsAndBytesConfig

BASE_SYSTEM_PROMPT = """You are MAGMA's robot commander.

You control a robot through optional function calls. Your job is to choose the next best response given:
- the current instruction or status update
- task attributes describing the current environment
- the full conversation history
- the available tools

Core decision rules:
- Use the history to infer what already happened, what failed, and what remains to do.
- Do not repeat a tool call that already succeeded.
- If a previous tool call failed, recover by correcting the call, choosing another valid tool, verifying state, or asking for missing information.
- Operate under partial observability: do not assume an object, location, or state exists unless it is in task attributes, memory, history, or can be checked with a tool.
- Maintain a coherent long-horizon plan, but only take the next necessary step.

Tool policy:
- Call a tool only when it is necessary for progress.
- Call at most one tool per response.
- Use only tools declared in the current tool list.
- Tool arguments must be valid JSON and must match the declared schema.
- Ground every argument in the provided context. Do not invent object names, robot names, quantities, or locations.
- Every tool call must select exactly one robot using a robot name from task attributes `known_robots`.
- If you call a tool, write exactly one tool block after the user-facing text:
- Correct tool-call format:
<tool_call>{"<robot_name>":{"name":"<func_name>","arguments":{"param":"value"}}}</tool_call>
- Never output the flat format {"name":"<func_name>","arguments":{...}}.
- If no tool is needed, do not output a tool block.

User-facing text policy:
- The text outside <tool_call> is the `say` response.
- Keep `say` short, operational, and consistent with any tool call.
- Do not write a `say:` label.
- Do not expose hidden reasoning, chain of thought, analysis, or <think> blocks.

Sometimes the user is just giving you rules and assignment, you must simply acknowledge without calling any tool. Be aware that tool execution (perception and action) are uncertain and may fail or give partial observation. Act in consequence.

Decision modes:
At each step, choose exactly one:
1. OBSERVE: gather missing or uncertain information using tools
2. ACT: execute one tool call
3. CLARIFY: ask the user for missing information

Belief tracking:
- Maintain a belief of the world based on the interaction (rules, assignment)
- If user ask to sort one or multiple object but you do not have any assignment in your history, ask for it.

Failure recovery:
If a tool fails:
- Do not retry blindly
- Consider possible causes:
  - missing object
  - wrong location
  - invalid arguments
  - execution failure
- Then:
  - verify with OBSERVE
  - try an alternative
  - or CLARIFY

Grounding constraint:
- Never invent objects, locations, or entities
- Use only information from attributes, history, or tool feedback

Planning:
- Focus on incremental, verifiable progress
- Avoid risky or assumption-heavy actions
- Re-check the environment when in doubt

Hint:
To make coffee you must place the mug, load the right capsule and start the machine.
To wash clothes you must put each requested clothe in the machine, then put the correct detergent (if you do not know witch detergent to use you must ask to the user) then start the machine.
Be aware that you can only have one object in your gripper. If you take an object, you need to put it somewhere before taking something else.

/no_think
"""


class QwenCommander(CausalModelClient):

    def __init__(self, settings: ModelSettings, name: str = "commander") -> None:
        super().__init__(settings, name)

    def _format_batch(self, message: BatchedMessageCommander) -> List[str]:
        system_message = {'role': "system", "content":BASE_SYSTEM_PROMPT}
        formatted_inputs = []
        self.input_elements = []
        instruction_roles = get_instruction_roles(message)
        batch_size = len(message.instruction)

        _validate_batch(message)

        for i in range(batch_size):
            
            mem_str = "Memory:\n"
            memory_list = get_memory_list(message.memory[i])
            if len(memory_list) > 0:
                for mem in memory_list:
                    mem_str += f"- {mem}\n"
            else:
                mem_str += "empty\n"

            prompt_user = f"Task Attributes : {message.attributes[i]}.\n{mem_str}\nQuery : {message.instruction[i]}"

            messages = [
                system_message.copy()
            ]
            for previous_mess in message.history[i]:
                role = map_chat_role(previous_mess.get("author"))
                messages.append({
                    "role": role,
                    "content": format_history_content(previous_mess, role),
                })

            messages.append({
                "role": map_chat_role(instruction_roles[i]),
                "content": prompt_user,
            })

            self.input_elements.append({"messages": messages, "tools": message.function[i],
                                        "enable_thinking": self.enable_thinking})
            formatted_inputs.append(
                _apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=message.function[i],
                    enable_thinking=self.enable_thinking,
                )
            )

        return formatted_inputs

    def process_batched_entry(self, message: BatchedMessageCommander, inference_mode: bool) -> List[Dict]:
        formatted_inputs = self._format_batch(message)
        inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(self.input_device)
        prompt_length = inputs["input_ids"].shape[1]
        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": self.use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            if inference_mode:
                output = self.model.generate(
                    **generation_kwargs,
                    do_sample=False,  
                )
            else:
                output = self.model.generate(
                    **generation_kwargs,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20
                )
        responses = []
        for i in range(len(formatted_inputs)):
            generated_tokens = output[i][prompt_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            parsed_response = parse_blocks(response_text)
            valid = (
                isinstance(parsed_response, dict)
                and isinstance(
                    parsed_response.get("action", {}),
                    (dict, list),
                )
            )
            self.log_prompt_exchange(
                formatted_inputs[i],
                response_text,
                valid,
            )
            responses.append(parsed_response)

        return responses


def _validate_batch(message: BatchedMessageCommander) -> None:
    batch_size = len(message.instruction)
    if not batch_size:
        raise ValueError("BatchedMessageCommander must contain at least one instruction.")

    for field_name in ("memory", "attributes", "history", "function"):
        field_value = getattr(message, field_name)
        if len(field_value) != batch_size:
            raise ValueError(
                f"{field_name} must have the same length as instruction "
                f"({len(field_value)} != {batch_size})."
            )


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
UNFINISHED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
UNFINISHED_TOOL_RE = re.compile(r"<tool_call>\s*(.*)$", re.DOTALL)


def _apply_chat_template(tokenizer: Any, messages: List[Dict[str, Any]], tools: Any, enable_thinking: bool) -> str:
    kwargs = {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as err:
        if "enable_thinking" not in str(err):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_blocks(text: str) -> Dict[str, Any]:
    raw_text = text.strip()
    text_without_think = THINK_RE.sub("", raw_text)
    text_without_think = UNFINISHED_THINK_RE.sub("", text_without_think).strip()

    tool_match = TOOL_RE.search(text_without_think) or UNFINISHED_TOOL_RE.search(text_without_think)
    action: Any = {}
    if tool_match:
        raw = tool_match.group(1).strip()
        parsed_action = _normalize_action(_load_json_object(raw, label="tool_call"))
        action = parsed_action or raw
    say = TOOL_RE.sub("", text_without_think).strip()
    say = UNFINISHED_TOOL_RE.sub("", say).strip()
    say = re.sub(r"^\s*say\s*:\s*", "", say, flags=re.IGNORECASE).strip()

    return {
        "think": "",
        "say": say,
        "action": action,
    }


def _load_json_object(text: Any, label: str = "JSON", quiet: bool = False) -> Dict[str, Any]:
    if not isinstance(text, str):
        return text if isinstance(text, dict) else {}
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        if not quiet:
            print(f"[PARSE ERROR] Invalid {label}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_action(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict) or not action:
        return {}
    if isinstance(action.get("name"), str):
        return {
            "name": action["name"],
            "arguments": action.get("arguments", {}) or {},
        }

    normalized_action = {}
    for value in action.values():
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            continue
        return {}

    for robot_name, value in action.items():
        normalized_action[robot_name] = {
            "name": value["name"],
            "arguments": value.get("arguments", {}) or {},
        }
    return normalized_action
 
