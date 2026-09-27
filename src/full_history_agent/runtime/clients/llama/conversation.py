import json
from typing import Any, Sequence

from magma_core.protocol.agent import AgentDecision

from ..history import get_history_content
from .prompt import build_system_prompt
from .serialization import assistant_message, decision_from_response


LLAMA_MEMORY_KEY = "_llama"
LLAMA_MEMORY_VERSION = 1


def clarification_turns(memory: dict[str, Any], history: Sequence[dict[str, Any]]) -> dict[str, str]:
    state = memory.get(LLAMA_MEMORY_KEY)
    if state is None:
        return {}
    if not isinstance(state, dict) or state.get("version") != LLAMA_MEMORY_VERSION:
        raise ValueError("memory._llama must be a version 1 object")
    turns = state.get("turns")
    if not isinstance(turns, dict):
        raise ValueError("memory._llama.turns must be an object")
    for index, kind in turns.items():
        if not isinstance(index, str) or not index.isdigit() or int(index) >= len(history):
            raise ValueError("Llama clarification index is outside memory.history")
        if kind != "clarification" or str(history[int(index)].get("author", "")).lower() not in {"model", "assistant"}:
            raise ValueError("Llama clarification must reference a model response")
    return dict(turns)


def format_current_input(instruction: str, attributes: dict[str, Any]) -> str:
    return (
        "Task attributes:\n"
        + json.dumps(attributes, ensure_ascii=False, sort_keys=True)
        + "\n\nCurrent input:\n"
        + instruction
    )


def build_messages(
    *,
    history: Sequence[dict[str, Any]],
    instruction: str,
    instruction_role: str,
    attributes: dict[str, Any],
    permanent_rules: Sequence[str],
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    turns = clarification_turns(memory, history)
    messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(permanent_rules)}]
    pending_kind: str | None = None
    for index, previous in enumerate(history):
        author = str(previous.get("author") or "USER").lower()
        content = get_history_content(previous)
        if author in {"model", "assistant"}:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and ({"say", "action"} & parsed.keys()):
                decision = decision_from_response(parsed)
            else:
                decision = AgentDecision(say=content)
            clarification = str(index) in turns
            messages.append(assistant_message(decision, clarification=clarification))
            pending_kind = "clarification" if clarification else ("tool_call" if decision.tool_calls else None)
            continue
        is_feedback = pending_kind and author in {"system", "status", "tool", "ipython"}
        is_clarification_answer = pending_kind == "clarification" and author == "user"
        if is_feedback or is_clarification_answer:
            messages.append({"role": "tool", "content": content})
            # Consecutive environment messages may contain individual results.
            if pending_kind == "clarification":
                pending_kind = None
            continue
        pending_kind = None
        if author in {"system", "status", "tool", "ipython"}:
            content = "Environment status:\n" + content
        messages.append({"role": "user", "content": content})

    current = format_current_input(instruction, attributes)
    if (pending_kind and instruction_role.lower() == "system") or pending_kind == "clarification":
        messages.append({"role": "tool", "content": current})
    else:
        if instruction_role.lower() == "system":
            current = "Environment status:\n" + current
        messages.append({"role": "user", "content": current})
    return messages


def update_clarification_memory(memory: dict[str, Any], response: dict[str, Any]) -> None:
    history = memory.get("history", [])
    turns = clarification_turns(memory, history)
    if response.get("_llama_kind") == "clarification":
        turns[str(len(history) + 1)] = "clarification"
    if turns:
        memory[LLAMA_MEMORY_KEY] = {"version": LLAMA_MEMORY_VERSION, "turns": turns}
