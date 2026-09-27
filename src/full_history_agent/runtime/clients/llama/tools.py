from typing import Any, Sequence

from ..qwen.tools import add_target_robot_parameter, normalize_tool_parameters


ASK_USER_TOOL = "ask_user"


def build_tool_descriptions(
    functions: Sequence[dict[str, Any]],
    attributes: dict[str, Any],
) -> list[dict[str, Any]]:
    known_robots = attributes.get("known_robots", [])
    robots = []
    if isinstance(known_robots, list):
        robots = [robot for robot in known_robots if isinstance(robot, str) and robot]
    tools = []
    for function in functions:
        parameters = normalize_tool_parameters(
            function.get("parameters", function.get("arguments")),
            function.get("optional", []),
        )
        tools.append({
            "type": "function",
            "function": {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": add_target_robot_parameter(parameters, robots),
            },
        })
    tools.append({
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL,
            "description": "Ask the user one concise question needed to continue the active task.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    })
    return tools
