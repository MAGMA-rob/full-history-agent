import json
from typing import Any

from magma_core.protocol.agent import AgentDecision, ToolCall

from .tools import ASK_USER_TOOL


def decision_from_response(response: dict[str, Any]) -> AgentDecision:
    """Convert MAGMA actions structurally, preserving order and duplicates."""
    action = response.get("action", {})
    actions = action if isinstance(action, list) else [action]
    calls = []
    for item in actions:
        if item == {}:
            continue
        if not isinstance(item, dict):
            raise ValueError("Actions must be JSON objects")
        if "name" in item:
            calls.append(ToolCall(
                name=item["name"],
                arguments=item.get("arguments", {}),
                target_robot_name=item.get("target_robot_name", item.get("target_robot")),
            ))
        else:
            for robot, call in item.items():
                if not isinstance(call, dict):
                    raise ValueError("Robot action must be an object")
                calls.append(ToolCall(
                    name=call.get("name"),
                    arguments=call.get("arguments", {}),
                    target_robot_name=robot,
                ))
    return AgentDecision(say=response.get("say", ""), tool_calls=calls)


def native_calls_from_decision(
    decision: AgentDecision,
    *,
    clarification: bool = False,
) -> list[dict[str, Any]]:
    if clarification:
        return [{"type": "function", "name": ASK_USER_TOOL, "parameters": {"question": decision.say}}]
    return [{
        "type": "function",
        "name": call.name,
        "parameters": {**call.arguments, "target_robot": call.target_robot_name},
    } for call in decision.tool_calls]


def serialize_decision(decision: AgentDecision, *, clarification: bool = False) -> str:
    """Render a native completion for inference history or an ideal target."""
    calls = native_calls_from_decision(decision, clarification=clarification)
    if not calls:
        return decision.say + "<|eot_id|>"
    payload = calls[0] if len(calls) == 1 else calls
    prefix = decision.say if not clarification else ""
    return (
        prefix + "<|python_tag|>"
        + json.dumps(payload, ensure_ascii=False, allow_nan=False)
        + "<|eom_id|>"
    )


def assistant_message(decision: AgentDecision, *, clarification: bool = False) -> dict[str, Any]:
    native_calls = native_calls_from_decision(decision, clarification=clarification)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if clarification else decision.say,
        "llama_completion": serialize_decision(decision, clarification=clarification),
    }
    if native_calls:
        message["tool_calls"] = [{
            "type": "function",
            "function": {"name": call["name"], "arguments": call["parameters"]},
        } for call in native_calls]
    return message
