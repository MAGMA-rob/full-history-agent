from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from .tools import ASK_USER_TOOL, ToolCatalog, actions_to_commander_format


TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
TERMINAL_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)+\s*$")


def parse_completion(
    *,
    tokenizer: Any,
    generated_ids: Sequence[int],
    model_type: str,
    catalog: ToolCatalog,
) -> Dict[str, Any]:
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    try:
        structured = _parse_with_response_schema(tokenizer, generated_ids)
        if structured is not None:
            content, reasoning, calls = structured
            if calls or "<tool_call>" not in content:
                return _response_from_parts(content, reasoning, calls, catalog)

        reasoning, content = _split_reasoning(TERMINAL_RE.sub("", text).strip())
        if not model_type.startswith("qwen2") and not model_type.startswith("qwen3"):
            raise ValueError(
                f"Unsupported Qwen model_type {model_type!r} without a response schema"
            )
        if re.search(r"<tool_call>\s*<function=", content):
            calls, visible = _parse_xml_calls(content, catalog)
        else:
            calls, visible = _parse_json_calls(content)
        return _response_from_parts(visible, reasoning, calls, catalog)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {
            "think": "",
            "say": "",
            "action": {},
            "_qwen_kind": "invalid",
            "_qwen_reasoning": "",
            "_qwen_valid": False,
            "_qwen_error": str(error),
        }


def _parse_with_response_schema(
    tokenizer: Any,
    generated_ids: Sequence[int],
) -> tuple[str, str, List[Dict[str, Any]]] | None:
    if not callable(getattr(tokenizer, "parse_response", None)):
        return None
    if getattr(tokenizer, "response_schema", None) is None:
        return None

    try:
        parsed = tokenizer.parse_response(list(generated_ids))
    except (AttributeError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        raise ValueError("Qwen response schema did not return an object")
    content = parsed.get("content", "") or ""
    reasoning = parsed.get("reasoning_content", parsed.get("thinking", "")) or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError("Qwen response content and reasoning must be strings")

    calls = []
    for raw_call in parsed.get("tool_calls", []) or []:
        if not isinstance(raw_call, dict):
            raise ValueError("Qwen response schema returned an invalid tool call")
        function = raw_call.get("function", raw_call)
        if not isinstance(function, dict):
            raise ValueError("Qwen response schema returned an invalid function call")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"name": function.get("name"), "arguments": arguments})
    return content.strip(), reasoning.strip(), calls


def _split_reasoning(text: str) -> tuple[str, str]:
    opening_count = text.count("<think>")
    closing_count = text.count("</think>")
    if opening_count > 1 or closing_count > 1:
        raise ValueError("Qwen response contains multiple thinking blocks")
    if opening_count == 1:
        opening = text.index("<think>")
        if closing_count != 1:
            raise ValueError("Qwen thinking block is not closed")
        closing = text.index("</think>", opening)
        prefix = text[:opening].strip()
        if prefix:
            raise ValueError("Unexpected content before Qwen thinking block")
        return (
            text[opening + len("<think>"):closing].strip(),
            text[closing + len("</think>"):].strip(),
        )
    if closing_count == 1:
        closing = text.index("</think>")
        return text[:closing].strip(), text[closing + len("</think>"):].strip()
    return "", text.strip()


def _parse_json_calls(content: str) -> tuple[List[Dict[str, Any]], str]:
    if content.count("<tool_call>") != content.count("</tool_call>"):
        raise ValueError("Qwen tool call block is not closed")
    matches = list(TOOL_BLOCK_RE.finditer(content))
    calls = []
    for match in matches:
        value = json.loads(match.group(1).strip())
        if not isinstance(value, dict):
            raise ValueError("Qwen tool call must contain a JSON object")
        calls.append({
            "name": value.get("name"),
            "arguments": value.get("arguments", {}),
        })
    visible = TOOL_BLOCK_RE.sub("", content).strip()
    return calls, visible


def _parse_xml_calls(
    content: str,
    catalog: ToolCatalog,
) -> tuple[List[Dict[str, Any]], str]:
    if content.count("<tool_call>") != content.count("</tool_call>"):
        raise ValueError("Qwen tool call block is not closed")
    matches = list(XML_CALL_RE.finditer(content))
    if "<tool_call>" in content and len(matches) != content.count("<tool_call>"):
        raise ValueError("Qwen3.5 tool call does not match the native XML format")

    calls = []
    for match in matches:
        name = match.group(1).strip()
        body = match.group(2)
        schema = catalog.parameters.get(name, {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        arguments: Dict[str, Any] = {}
        parameter_matches = list(XML_PARAMETER_RE.finditer(body))
        if XML_PARAMETER_RE.sub("", body).strip():
            raise ValueError(f"Qwen3.5 tool {name!r} contains invalid parameter markup")
        for parameter in parameter_matches:
            key = parameter.group(1).strip()
            if key in arguments:
                raise ValueError(f"Qwen3.5 tool {name!r} repeats parameter {key!r}")
            arguments[key] = _coerce_xml_value(
                parameter.group(2).strip(),
                properties.get(key, {}),
            )
        calls.append({"name": name, "arguments": arguments})

    visible = XML_CALL_RE.sub("", content).strip()
    return calls, visible


def _coerce_xml_value(value: str, schema: Any) -> Any:
    value_type = schema.get("type") if isinstance(schema, dict) else None
    if value_type == "string" or value_type is None:
        return value
    if value_type == "integer":
        return int(value)
    if value_type == "number":
        return float(value)
    if value_type == "boolean":
        if value.lower() not in {"true", "false"}:
            raise ValueError(f"Invalid boolean tool argument {value!r}")
        return value.lower() == "true"
    if value_type in {"array", "object"}:
        parsed = json.loads(value)
        expected = list if value_type == "array" else dict
        if not isinstance(parsed, expected):
            raise ValueError(f"Tool argument does not match type {value_type!r}")
        return parsed
    return value


def _response_from_parts(
    content: str,
    reasoning: str,
    calls: Sequence[Dict[str, Any]],
    catalog: ToolCatalog,
) -> Dict[str, Any]:
    if calls and content.strip():
        raise ValueError("Qwen response cannot combine user-facing text and tool calls")
    if not calls:
        if not content.strip():
            raise ValueError("Qwen response contains neither a final answer nor a tool call")
        return _response(
            say=content.strip(),
            action={},
            kind="final",
            reasoning="",
        )

    normalized_calls = [_validate_call(call, catalog) for call in calls]
    clarification_calls = [
        call for call in normalized_calls if call["name"] == ASK_USER_TOOL
    ]
    if clarification_calls:
        if len(normalized_calls) != 1:
            raise ValueError("ask_user cannot be combined with another tool call")
        return _response(
            say=clarification_calls[0]["arguments"]["question"],
            action={},
            kind="clarification",
            reasoning=reasoning,
        )

    robots = [call["target_robot"] for call in normalized_calls]
    if len(robots) != len(set(robots)):
        raise ValueError("Parallel Qwen calls may target each robot at most once")
    return _response(
        say="",
        action=actions_to_commander_format(normalized_calls),
        kind="tool_call",
        reasoning=reasoning,
    )


def _validate_call(call: Dict[str, Any], catalog: ToolCatalog) -> Dict[str, Any]:
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or name not in catalog.names:
        raise ValueError(f"Unknown Qwen tool name {name!r}")
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for Qwen tool {name!r} must be an object")

    schema = catalog.parameters[name]
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ValueError(f"Qwen tool {name!r} is missing required arguments: {missing}")
    if schema.get("additionalProperties") is False:
        unknown = set(arguments) - set(schema.get("properties", {}))
        if unknown:
            raise ValueError(f"Qwen tool {name!r} has unknown arguments: {sorted(unknown)}")

    if name == ASK_USER_TOOL:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("ask_user requires a non-empty question")
        return {"name": name, "arguments": {"question": question.strip()}}

    target_robot = arguments.get("target_robot")
    if not isinstance(target_robot, str) or target_robot not in catalog.known_robots:
        raise ValueError(f"Unknown target robot {target_robot!r} for Qwen tool {name!r}")
    environment_arguments = dict(arguments)
    environment_arguments.pop("target_robot")
    return {
        "name": name,
        "arguments": environment_arguments,
        "target_robot": target_robot,
    }


def _response(
    *,
    say: str,
    action: Any,
    kind: str,
    reasoning: str,
) -> Dict[str, Any]:
    return {
        "think": "",
        "say": say,
        "action": action,
        "_qwen_kind": kind,
        "_qwen_reasoning": reasoning,
        "_qwen_valid": True,
    }
