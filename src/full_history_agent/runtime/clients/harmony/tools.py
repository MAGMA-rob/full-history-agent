from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Sequence

from openai_harmony import Message, Role, ToolDescription


ASK_USER_TOOL = "ask_user"
PARALLEL_TOOL = "execute_parallel"
RESERVED_TOOL_NAMES = {ASK_USER_TOOL, PARALLEL_TOOL}


@dataclass(frozen=True)
class ToolCatalog:
    names: frozenset[str]
    known_robots: tuple[str, ...]


def build_tool_descriptions(
    functions: Sequence[Dict[str, Any]],
    attributes: Any,
) -> tuple[List[ToolDescription], ToolCatalog]:
    known_robots = _known_robot_names(attributes)
    descriptions: List[ToolDescription] = []
    normalized_tools: List[tuple[str, str, Dict[str, Any]]] = []

    for tool in functions:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool declaration is missing a name: {tool}")
        if name in RESERVED_TOOL_NAMES:
            raise ValueError(f"Tool name {name!r} is reserved by the GPT-OSS adapter")

        parameters = normalize_tool_parameters(
            tool.get("parameters", tool.get("arguments")),
            tool.get("optional", []),
        )
        normalized_tools.append((name, str(tool.get("description", "")), parameters))
        descriptions.append(ToolDescription.new(
            name,
            str(tool.get("description", "")),
            parameters=add_target_robot_parameter(parameters, known_robots),
        ))

    descriptions.append(ToolDescription.new(
        ASK_USER_TOOL,
        "Ask the user one concise question for information required to continue the active task.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The exact concise question to show to the user.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ))

    if normalized_tools and len(known_robots) >= 2:
        variants = []
        for name, description, parameters in normalized_tools:
            variants.append({
                "type": "object",
                "description": description,
                "properties": {
                    "target_robot": {
                        "type": "string",
                        "enum": known_robots,
                    },
                    "name": {
                        "type": "string",
                        "enum": [name],
                    },
                    "arguments": deepcopy(parameters),
                },
                "required": ["target_robot", "name", "arguments"],
                "additionalProperties": False,
            })

        descriptions.append(ToolDescription.new(
            PARALLEL_TOOL,
            "Start two or more independent environment actions together. Each robot may appear at most once.",
            parameters={
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": len(known_robots),
                        "items": {"oneOf": variants},
                    }
                },
                "required": ["calls"],
                "additionalProperties": False,
            },
        ))

    return descriptions, ToolCatalog(
        names=frozenset(name for name, _, _ in normalized_tools),
        known_robots=tuple(known_robots),
    )


def tool_message_from_actions(actions: Sequence[Dict[str, Any]]) -> Message:
    if not actions:
        raise ValueError("Cannot build a Harmony tool call without an action")

    validate_actions(actions)
    if len(actions) == 1:
        action = actions[0]
        arguments = dict(action.get("arguments", {}) or {})
        arguments["target_robot"] = action["target_robot"]
        recipient = f"functions.{action['name']}"
        content = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
    else:
        recipient = f"functions.{PARALLEL_TOOL}"
        content = json.dumps({"calls": list(actions)}, ensure_ascii=True, sort_keys=True)

    return (
        Message.from_role_and_content(Role.ASSISTANT, content)
        .with_channel("commentary")
        .with_recipient(recipient)
        .with_content_type("json")
    )


def recipient_for_actions(actions: Sequence[Dict[str, Any]]) -> str:
    if len(actions) == 1:
        return f"functions.{actions[0]['name']}"
    if len(actions) > 1:
        return f"functions.{PARALLEL_TOOL}"
    raise ValueError("Cannot resolve a recipient without actions")


def parse_tool_call(
    recipient: str,
    text: str,
    catalog: ToolCatalog,
) -> tuple[str, Any]:
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Tool call arguments must be valid JSON") from error
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object")

    name = recipient.rsplit(".", 1)[-1]
    if recipient == f"functions.{ASK_USER_TOOL}":
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("ask_user requires a non-empty question")
        return "clarification", question.strip()

    if recipient == f"functions.{PARALLEL_TOOL}":
        calls = arguments.get("calls")
        if not isinstance(calls, list) or len(calls) < 2:
            raise ValueError("execute_parallel requires at least two calls")
        actions = [_normalize_parsed_action(call, catalog) for call in calls]
        validate_actions(actions, catalog)
        return "tool_call", actions

    if not recipient.startswith("functions.") or name not in catalog.names:
        raise ValueError(f"Unknown tool recipient {recipient!r}")

    target_robot = arguments.pop("target_robot", None)
    action = {
        "name": name,
        "arguments": arguments,
        "target_robot": target_robot,
    }
    validate_actions([action], catalog)
    return "tool_call", [action]


def validate_actions(
    actions: Sequence[Dict[str, Any]],
    catalog: ToolCatalog | None = None,
) -> None:
    robots = []
    for action in actions:
        name = action.get("name")
        robot = action.get("target_robot")
        arguments = action.get("arguments", {})
        if not isinstance(name, str) or not name or name in RESERVED_TOOL_NAMES:
            raise ValueError(f"Invalid environment tool name: {name!r}")
        if not isinstance(robot, str) or not robot:
            raise ValueError(f"Tool {name!r} requires a target_robot")
        if not isinstance(arguments, dict):
            raise ValueError(f"Arguments for tool {name!r} must be an object")
        if catalog is not None:
            if name not in catalog.names:
                raise ValueError(f"Unknown environment tool {name!r}")
            if robot not in catalog.known_robots:
                raise ValueError(f"Unknown target robot {robot!r}")
        robots.append(robot)

    if len(robots) != len(set(robots)):
        raise ValueError("A parallel response may contain at most one call per robot")


def normalize_actions(action: Any) -> List[Dict[str, Any]]:
    if not action:
        return []
    if isinstance(action, list):
        normalized: List[Dict[str, Any]] = []
        for item in action:
            normalized.extend(normalize_actions(item))
        return normalized
    if not isinstance(action, dict):
        return []
    if isinstance(action.get("name"), str):
        target_robot = action.get("target_robot", action.get("target_robot_name"))
        return [{
            "name": action["name"],
            "arguments": action.get("arguments", {}) or {},
            "target_robot": target_robot,
        }]

    normalized = []
    for robot_name, value in action.items():
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            normalized.append({
                "name": value["name"],
                "arguments": value.get("arguments", {}) or {},
                "target_robot": robot_name,
            })
    return normalized


def actions_to_commander_format(actions: Sequence[Dict[str, Any]]) -> Any:
    wrapped = [
        {
            action["target_robot"]: {
                "name": action["name"],
                "arguments": action.get("arguments", {}),
            }
        }
        for action in actions
    ]
    return wrapped[0] if len(wrapped) == 1 else wrapped


def normalize_tool_parameters(
    parameters: Any,
    optional: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if parameters is None:
        return {"type": "object", "properties": {}}
    if isinstance(parameters, dict) and parameters.get("type") == "object":
        schema = deepcopy(parameters)
        schema.setdefault("properties", {})
        return schema
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    optional_names = set(optional or [])
    properties: Dict[str, Any] = {}
    required = []
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            properties[name] = {"type": _json_schema_type(spec)}
        else:
            prop = deepcopy(spec)
            prop["type"] = _json_schema_type(prop.get("type", "string"))
            properties[name] = prop
        if name not in optional_names:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def add_target_robot_parameter(
    parameters: Dict[str, Any],
    known_robots: Sequence[str],
) -> Dict[str, Any]:
    schema = deepcopy(parameters)
    properties = dict(schema.get("properties", {}))
    target_schema: Dict[str, Any] = {
        "type": "string",
        "description": "Robot that must execute this tool call.",
    }
    if known_robots:
        target_schema["enum"] = list(known_robots)
    properties["target_robot"] = target_schema
    schema["properties"] = properties
    required = list(schema.get("required", []))
    if "target_robot" not in required:
        required.append("target_robot")
    schema["required"] = required
    return schema


def _normalize_parsed_action(value: Any, catalog: ToolCatalog) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each parallel call must be an object")
    action = {
        "name": value.get("name"),
        "arguments": value.get("arguments", {}),
        "target_robot": value.get("target_robot"),
    }
    validate_actions([action], catalog)
    return action


def _known_robot_names(attributes: Any) -> List[str]:
    if not isinstance(attributes, dict):
        return []
    known_robots = attributes.get("known_robots", [])
    if not isinstance(known_robots, list):
        return []
    return [robot for robot in known_robots if isinstance(robot, str) and robot]


def _json_schema_type(value: Any) -> str:
    if isinstance(value, type):
        value = value.__name__
    return {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }.get(str(value).lower(), "string")
