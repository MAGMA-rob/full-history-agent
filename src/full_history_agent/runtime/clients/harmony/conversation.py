from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
)

from ..history import get_history_content
from .prompt import build_developer_instructions
from .tools import (
    ASK_USER_TOOL,
    normalize_actions,
    recipient_for_actions,
    tool_message_from_actions,
)


GPT_OSS_MEMORY_KEY = "_gpt_oss"
GPT_OSS_MEMORY_VERSION = 1


def history_window_start(
    history: Sequence[Dict[str, Any]],
    history_max_messages: int | None,
) -> int:
    if history_max_messages is None:
        return 0
    if type(history_max_messages) is not int or history_max_messages <= 0:
        raise ValueError("history_max_messages must be a positive integer or None")
    return max(0, len(history) - history_max_messages)


def build_conversation(
    *,
    history: Sequence[Dict[str, Any]],
    instruction: str,
    instruction_role: str,
    attributes: Any,
    permanent_rules: Sequence[Any],
    tools: Sequence[ToolDescription],
    reasoning_effort: ReasoningEffort,
    memory: Dict[str, Any],
    history_max_messages: int | None = None,
) -> Conversation:
    turns = validate_analysis_state(memory, history)
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_required_channels(["analysis", "commentary", "final"])
    )
    developer_content = DeveloperContent.new().with_instructions(
        build_developer_instructions(permanent_rules)
    )
    if tools:
        developer_content = developer_content.with_function_tools(tools)

    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.DEVELOPER, developer_content),
    ]
    pending_recipient: Optional[str] = None

    start_index = history_window_start(history, history_max_messages)
    for index, previous_message in enumerate(history[start_index:], start=start_index):
        author = previous_message.get("author")
        normalized_author = (author or "USER").lower()
        content = get_history_content(previous_message)

        if normalized_author in ("model", "assistant"):
            record = turns.get(str(index))
            if record is not None:
                for analysis in record["analysis"]:
                    messages.append(
                        Message.from_role_and_content(Role.ASSISTANT, analysis)
                        .with_channel("analysis")
                    )

            say, actions = split_model_history_content(content)
            if record is not None and record["kind"] == "clarification":
                if not say:
                    raise ValueError(
                        f"GPT-OSS clarification at history index {index} has no question"
                    )
                messages.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT,
                        json.dumps({"question": say}, ensure_ascii=True, sort_keys=True),
                    )
                    .with_channel("commentary")
                    .with_recipient(f"functions.{ASK_USER_TOOL}")
                    .with_content_type("json")
                )
                pending_recipient = f"functions.{ASK_USER_TOOL}"
                continue

            if say:
                messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, say).with_channel("final")
                )
                pending_recipient = None
            if actions:
                tool_message = tool_message_from_actions(actions)
                messages.append(tool_message)
                pending_recipient = recipient_for_actions(actions)
            continue

        if normalized_author in ("system", "status") and pending_recipient:
            messages.append(tool_result_message(
                pending_recipient,
                strip_previous_tool_call(content),
            ))
            pending_recipient = None
            continue

        if normalized_author in ("user", "") and pending_recipient:
            if pending_recipient == f"functions.{ASK_USER_TOOL}":
                messages.append(tool_result_message(pending_recipient, content))
                pending_recipient = None
                continue

            messages.append(tool_result_message(
                pending_recipient,
                json.dumps({"status": "completed"}, ensure_ascii=True),
            ))
            pending_recipient = None

        messages.append(history_message(author, content))

    current_content = format_current_input(instruction, attributes)
    normalized_current_role = instruction_role.lower()
    if pending_recipient and normalized_current_role == "system":
        messages.append(tool_result_message(
            pending_recipient,
            format_current_input(strip_previous_tool_call(instruction), attributes),
        ))
    elif pending_recipient == f"functions.{ASK_USER_TOOL}":
        messages.append(tool_result_message(pending_recipient, current_content))
    else:
        if pending_recipient:
            messages.append(tool_result_message(
                pending_recipient,
                json.dumps({"status": "completed"}, ensure_ascii=True),
            ))
        messages.append(current_message(instruction_role, current_content))

    return Conversation.from_messages(messages)


def update_analysis_memory(memory: Dict[str, Any], response: Dict[str, Any]) -> None:
    kind = response.get("_gpt_oss_kind")
    if kind == "final" or (response.get("say") and kind != "clarification"):
        memory.pop(GPT_OSS_MEMORY_KEY, None)
        return
    if kind not in {"tool_call", "clarification"}:
        return

    history = memory.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("memory.history must be a list before updating GPT-OSS state")
    turns = validate_analysis_state(memory, history)
    model_history_index = len(history) + 1
    turns[str(model_history_index)] = {
        "kind": kind,
        "analysis": list(response.get("_gpt_oss_analysis", [])),
    }
    memory[GPT_OSS_MEMORY_KEY] = {
        "version": GPT_OSS_MEMORY_VERSION,
        "turns": turns,
    }


def validate_analysis_state(
    memory: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    state = memory.get(GPT_OSS_MEMORY_KEY)
    if state is None:
        return {}
    if not isinstance(state, dict) or state.get("version") != GPT_OSS_MEMORY_VERSION:
        raise ValueError("memory._gpt_oss must be a version 1 object")
    turns = state.get("turns")
    if not isinstance(turns, dict):
        raise ValueError("memory._gpt_oss.turns must be an object")

    validated: Dict[str, Dict[str, Any]] = {}
    for raw_index, record in turns.items():
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            raise ValueError("GPT-OSS turn keys must be decimal history indexes")
        index = int(raw_index)
        if index >= len(history):
            raise ValueError(f"GPT-OSS turn index {index} is outside memory.history")
        author = str(history[index].get("author", "")).lower()
        if author not in {"model", "assistant"}:
            raise ValueError(f"GPT-OSS turn index {index} does not reference a model response")
        if not isinstance(record, dict) or record.get("kind") not in {
            "tool_call", "clarification",
        }:
            raise ValueError(f"Invalid GPT-OSS turn record at history index {index}")
        analysis = record.get("analysis")
        if not isinstance(analysis, list) or any(not isinstance(item, str) for item in analysis):
            raise ValueError(f"GPT-OSS analysis at history index {index} must be a list of strings")
        validated[raw_index] = {
            "kind": record["kind"],
            "analysis": list(analysis),
        }
    return validated


def format_current_input(instruction: str, attributes: Any) -> str:
    return (
        "Task attributes:\n"
        f"{json.dumps(attributes, ensure_ascii=True, sort_keys=True)}\n\n"
        "Current input:\n"
        f"{instruction}"
    )


def history_message(author: Optional[str], content: str) -> Message:
    normalized_author = (author or "USER").lower()
    if normalized_author in ("model", "assistant"):
        return Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
    if normalized_author in ("system", "status"):
        content = f"Status update:\n{content}"
    return Message.from_role_and_content(Role.USER, content)


def current_message(author: str, content: str) -> Message:
    if author.lower() == "system":
        content = f"Environment status:\n{content}"
    return Message.from_role_and_content(Role.USER, content)


def tool_result_message(recipient: str, content: str) -> Message:
    return (
        Message.from_author_and_content(Author.new(Role.TOOL, recipient), content)
        .with_channel("commentary")
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
