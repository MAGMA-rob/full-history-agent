from copy import deepcopy
from typing import Any

from ..structure import HRActionProjection


class HRActionProjector:
    """Serialize one resolved HR action without mutating replay state."""

    def project(self, projection: HRActionProjection) -> dict[str, Any]:
        if (projection.calls is None) == (projection.message is None):
            raise ValueError(
                "An HR projection requires exactly one action kind."
            )

        if projection.message is not None:
            target: dict[str, Any] = {
                "say": projection.message,
                "action": {},
            }
        else:
            actions = [{call.target_robot_name: {'name': call.name, 'arguments': deepcopy(call.arguments)}}
                       for call in projection.calls or ()]
            flattened = {robot: value for action in actions for robot, value in action.items()}
            target = {'say': '', 'action': flattened if len(flattened) == len(actions) else actions}


        return {
            "input": {
                "memory": {},
                "attributes": deepcopy(projection.attributes),
                "instruction": projection.instruction.content,
                "instruction_role": projection.instruction.author,
                "history": [
                    {
                        "author": entry.author,
                        "content": entry.content,
                        "timestamp": 0,
                    }
                    for entry in projection.history
                ],
            },
            "target": target,
        }
