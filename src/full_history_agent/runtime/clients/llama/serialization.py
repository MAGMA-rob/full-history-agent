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


def assistant_message(decision: AgentDecision, *, clarification: bool = False) -> dict[str, Any]:
    """Render tool history in the same format requested from the model."""
    if clarification:
        calls = [{"name": ASK_USER_TOOL, "parameters": {"question": decision.say}}]
    else:
        calls = [{
            "name": call.name,
            "parameters": {**call.arguments, "target_robot": call.target_robot_name},
        } for call in decision.tool_calls]

    if not calls:
        return {"role": "assistant", "content": decision.say}

    payload = "; ".join(json.dumps(call, ensure_ascii=False, allow_nan=False) for call in calls)
    prefix = "" if clarification else decision.say
    return {
        "role": "assistant",
        "content": prefix + "<|python_tag|>" + payload + "<|eom_id|>",
    }
