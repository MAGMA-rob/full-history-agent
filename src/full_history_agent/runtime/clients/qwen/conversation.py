from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..history import get_history_content
from .prompt import build_system_prompt
from .tools import ASK_USER_TOOL, normalize_actions, tool_calls_from_actions


QWEN_MEMORY_KEY = "_qwen"
QWEN_MEMORY_VERSION = 1


def build_messages(
    *,
    history: Sequence[Dict[str, Any]],
    instruction: str,
    instruction_role: str,
    attributes: Any,
    permanent_rules: Sequence[Any],
    memory: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool]:
    turns = validate_reasoning_state(memory, history)
    messages: List[Dict[str, Any]] = [{
        "role": "system",
        "content": build_system_prompt(permanent_rules),
    }]
    pending_kind: Optional[str] = None

    for index, previous_message in enumerate(history):
        author = str(previous_message.get("author") or "USER").lower()
        content = get_history_content(previous_message)

        if author in {"model", "assistant"}:
            say, actions = split_model_history_content(content)
            record = turns.get(str(index))
            if record is not None and record["kind"] == "clarification":
                if not say:
                    raise ValueError(
                        f"Qwen clarification at history index {index} has no question"
                    )
                assistant = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": ASK_USER_TOOL,
                            "arguments": {"question": say},
                        },
                    }],
                }
                if record["reasoning"]:
                    assistant["reasoning_content"] = record["reasoning"]
                messages.append(assistant)
                pending_kind = "clarification"
                continue

            if actions:
                assistant = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls_from_actions(actions),
                }
                if record is not None and record["reasoning"]:
                    assistant["reasoning_content"] = record["reasoning"]
                messages.append(assistant)
                pending_kind = "tool_call"
                continue

            messages.append({"role": "assistant", "content": say})
            pending_kind = None
            continue

        if author in {"system", "status"} and pending_kind:
            messages.append({
                "role": "tool",
                "content": strip_previous_tool_call(content),
            })
            pending_kind = None
            continue

        if author in {"user", ""} and pending_kind == "clarification":
            messages.append({"role": "tool", "content": content})
            pending_kind = None
            continue

        if pending_kind:
            messages.append({
                "role": "tool",
                "content": json.dumps({"status": "completed"}, ensure_ascii=True),
            })
            pending_kind = None

        if author in {"system", "status"}:
            content = f"Environment status:\n{content}"
        messages.append({"role": "user", "content": content})

    current_content = format_current_input(instruction, attributes)
    normalized_role = instruction_role.lower()
    starts_new_task = normalized_role == "user" and pending_kind != "clarification"
    if pending_kind and normalized_role == "system":
        messages.append({
            "role": "tool",
            "content": format_current_input(
                strip_previous_tool_call(instruction),
                attributes,
            ),
        })
    elif pending_kind == "clarification":
        messages.append({"role": "tool", "content": current_content})
        starts_new_task = False
    else:
        if pending_kind:
            messages.append({
                "role": "tool",
                "content": json.dumps({"status": "completed"}, ensure_ascii=True),
            })
        if normalized_role == "system":
            current_content = f"Environment status:\n{current_content}"
        messages.append({"role": "user", "content": current_content})

    return messages, starts_new_task


def update_reasoning_memory(memory: Dict[str, Any], response: Dict[str, Any]) -> None:
    kind = response.get("_qwen_kind")
    if kind == "final":
        memory.pop(QWEN_MEMORY_KEY, None)
        return
    if kind not in {"tool_call", "clarification"}:
        return

    history = memory.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("memory.history must be a list before updating Qwen state")
    turns = (
        {}
        if response.get("_qwen_new_task")
        else validate_reasoning_state(memory, history)
    )
    model_history_index = len(history) + 1
    turns[str(model_history_index)] = {
        "kind": kind,
        "reasoning": str(response.get("_qwen_reasoning", "") or ""),
    }
    memory[QWEN_MEMORY_KEY] = {
        "version": QWEN_MEMORY_VERSION,
        "turns": turns,
    }


def validate_reasoning_state(
    memory: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    state = memory.get(QWEN_MEMORY_KEY)
    if state is None:
        return {}
    if not isinstance(state, dict) or state.get("version") != QWEN_MEMORY_VERSION:
        raise ValueError("memory._qwen must be a version 1 object")
    turns = state.get("turns")
    if not isinstance(turns, dict):
        raise ValueError("memory._qwen.turns must be an object")

    validated: Dict[str, Dict[str, str]] = {}
    for raw_index, record in turns.items():
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            raise ValueError("Qwen turn keys must be decimal history indexes")
        index = int(raw_index)
        if index >= len(history):
            raise ValueError(f"Qwen turn index {index} is outside memory.history")
        author = str(history[index].get("author", "")).lower()
        if author not in {"model", "assistant"}:
            raise ValueError(f"Qwen turn index {index} does not reference a model response")
        if not isinstance(record, dict) or record.get("kind") not in {
            "tool_call",
            "clarification",
        }:
            raise ValueError(f"Invalid Qwen turn record at history index {index}")
        reasoning = record.get("reasoning")
        if not isinstance(reasoning, str):
            raise ValueError(f"Qwen reasoning at history index {index} must be a string")
        validated[raw_index] = {
            "kind": record["kind"],
            "reasoning": reasoning,
        }
    return validated


def format_current_input(instruction: str, attributes: Any) -> str:
    return (
        "Task attributes:\n"
        f"{json.dumps(attributes, ensure_ascii=True, sort_keys=True)}\n\n"
        "Current input:\n"
        f"{instruction}"
    )


def split_model_history_content(content: str) -> tuple[str, List[Dict[str, Any]]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip(), []
    if not isinstance(parsed, dict):
        return content.strip(), []
    if "say" in parsed or "action" in parsed:
        return str(parsed.get("say", "") or ""), normalize_actions(parsed.get("action"))
    actions = normalize_actions(parsed)
    return ("", actions) if actions else (content.strip(), [])


def strip_previous_tool_call(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict):
        parsed.pop("previous_tool_call", None)
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True)
    return content
