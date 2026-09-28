import json
from typing import Any

from magma_core.protocol.agent import AgentDecision, ToolCall

from .tools import ASK_USER_TOOL


TERMINAL_TOKENS = ("<|eom_id|>", "<|eot_id|>", "<|end_of_text|>")


def invalid_response(error: Exception | str) -> dict[str, Any]:
    return {
        "say": "", "action": {}, "_llama_kind": "invalid",
        "_llama_valid": False, "_llama_error": str(error) or "Invalid Llama output",
    }


def parse_completion(text: str) -> dict[str, Any]:
    """Parse a whole completion; semantic tool validation belongs downstream."""
    try:
        content = text.strip()
        terminal = next((token for token in TERMINAL_TOKENS if content.endswith(token)), None)
        if terminal is None:
            return invalid_response("Llama completion has no terminal token (possibly truncated)")
        content = content[:-len(terminal)].strip()
        if not content:
            return invalid_response("Llama completion is empty")
        marker = "<|python_tag|>"
        if marker not in content:
            if terminal == "<|eom_id|>":
                return invalid_response("Llama tool request has no python_tag")
            return {"say": content, "action": {}, "_llama_kind": "final", "_llama_valid": True}

        visible, payload = content.split(marker, 1)
        decoder = json.JSONDecoder()
        native_calls: list[dict[str, Any]] = []
        offset = 0
        while offset < len(payload):
            while offset < len(payload) and payload[offset].isspace():
                offset += 1
            if offset == len(payload):
                break
            native, offset = decoder.raw_decode(payload, offset)
            if not isinstance(native, dict):
                return invalid_response("Llama tool call must be a JSON object")
            native_calls.append(native)
            while offset < len(payload) and payload[offset].isspace():
                offset += 1
            if offset < len(payload):
                if payload[offset] != ";":
                    return invalid_response("Llama tool calls must be separated by semicolons")
                offset += 1
                if not payload[offset:].strip():
                    return invalid_response("Llama tool call sequence ends with a semicolon")
        if not native_calls:
            return invalid_response("Llama tool request is empty")
        calls = []
        question = None
        for native in native_calls:
            if not isinstance(native, dict):
                return invalid_response("Llama tool call must be a JSON object")
            name = native.get("name")
            parameters = native.get("parameters")
            if not isinstance(parameters, dict):
                return invalid_response("Llama tool parameters must be a JSON object")
            if name == ASK_USER_TOOL:
                question = parameters.get("question")
                if len(native_calls) != 1 or not isinstance(question, str) or not question.strip():
                    return invalid_response("ask_user requires one standalone call with a nonempty question")
                continue
            arguments = dict(parameters)
            robot = arguments.pop("target_robot", None)
            calls.append(ToolCall(name=name, arguments=arguments, target_robot_name=robot))
        if question is not None:
            return {
                "say": visible + question, "action": {},
                "_llama_kind": "clarification", "_llama_valid": True,
            }
        decision = AgentDecision(say=visible, tool_calls=calls)
        return {
            "say": decision.say,
            "action": [{call.target_robot_name: {"name": call.name, "arguments": call.arguments}}
                       for call in decision.tool_calls],
            "_llama_kind": "tool_call", "_llama_valid": True,
        }
    except Exception as error:
        # A malformed candidate must never interrupt its batch or HTTP request.
        return invalid_response(error)
