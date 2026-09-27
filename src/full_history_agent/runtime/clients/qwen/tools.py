from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


ASK_USER_TOOL = "ask_user"


@dataclass(frozen=True)
class ToolCatalog:
    names: frozenset[str]
    known_robots: tuple[str, ...]
    parameters: Dict[str, Dict[str, Any]]


def build_tool_descriptions(
    functions: Sequence[Dict[str, Any]],
    attributes: Any,
) -> tuple[List[Dict[str, Any]], ToolCatalog]:
    known_robots = _known_robot_names(attributes)
    tools: List[Dict[str, Any]] = []
    parameters_by_name: Dict[str, Dict[str, Any]] = {}

    for tool in functions:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool declaration is missing a name: {tool}")
        if name == ASK_USER_TOOL:
            raise ValueError(f"Tool name {name!r} is reserved by the Qwen adapter")
        if name in parameters_by_name:
            raise ValueError(f"Tool name {name!r} is declared more than once")

        parameters = normalize_tool_parameters(
            tool.get("parameters", tool.get("arguments")),
            tool.get("optional", []),
        )
        parameters = add_target_robot_parameter(parameters, known_robots)
        parameters_by_name[name] = parameters
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description", "")),
                "parameters": parameters,
            },
        })

    ask_user_parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The exact concise question to show to the user.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    parameters_by_name[ASK_USER_TOOL] = ask_user_parameters
    tools.append({
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL,
            "description": (
                "Ask the user one concise question for information required to "
                "continue the active task."
            ),
            "parameters": ask_user_parameters,
        },
    })

    return tools, ToolCatalog(
        names=frozenset(parameters_by_name),
        known_robots=tuple(known_robots),
        parameters=parameters_by_name,
    )


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
        return [{
            "name": action["name"],
            "arguments": action.get("arguments", {}) or {},
            "target_robot": action.get("target_robot", action.get("target_robot_name")),
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


def tool_calls_from_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_calls = []
    for action in actions:
        arguments = dict(action.get("arguments", {}) or {})
        arguments["target_robot"] = action.get("target_robot")
        tool_calls.append({
            "type": "function",
            "function": {
                "name": action.get("name"),
                "arguments": arguments,
            },
        })
    return tool_calls


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
