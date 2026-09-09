from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from magma_offpolicy_gen.traces import MessageEvent, ToolCallEvent, ToolResultEvent, TraceEvent, UserEvent

from magma_offpolicy_gen.agents.base import (
    AgentGenerationResult,
    BaseAgentFormat,
    ExportOptions,
    ExportSummary,
    declared_known_robots,
)
from magma_offpolicy_gen.agents.execution_context import TraceExecutionContext
from magma_offpolicy_gen.agents.metadata import build_agent_metadata
from magma_offpolicy_gen.generation.models import GeneratedTask
from .generators import HRActionProjector
from .structure import (
    HRActionProjection,
    HRHistoryEntry,
    HRInstruction,
    HRProjectionContext,
)

if TYPE_CHECKING:
    from magma_core.simulation.tasks import TaskDefinition


class HRAgentFormat(BaseAgentFormat):
    """Replay linear PerfectTrace events into History Reactive examples."""

    name = "hr"
    requires_export_backends = True
    allows_export_without_backends = True

    def __init__(self) -> None:
        self.action_projector = HRActionProjector()

    def _process_event(
        self,
        context: HRProjectionContext,
        event: TraceEvent,
    ) -> HRActionProjection | None:
        if isinstance(event, UserEvent):
            context.set_instruction(HRInstruction(
                author="USER",
                content=event.content,
            ))
            return None

        if isinstance(event, ToolResultEvent):
            messages = context.execution.receive_tool_results(event.results)
            infos: str | list[str] = (
                messages[0] if len(messages) == 1 else list(messages)
            )
            context.set_instruction(HRInstruction(
                author="SYSTEM",
                content=json.dumps(
                    {"infos": infos},
                    ensure_ascii=False,
                ),
            ))
            return None

        if not isinstance(event, (ToolCallEvent, MessageEvent)):
            return None

        instruction = context.pending_instruction
        if instruction is None:
            raise RuntimeError(
                f"A {type(event).__name__} requires a pending instruction."
            )

        projection = HRActionProjection(
            attributes=context.execution.attributes_view(
                include_robot_statuses=True
            ),
            instruction=instruction,
            history=tuple(context.history),
            calls=(
                tuple(event.calls)
                if isinstance(event, ToolCallEvent)
                else None
            ),
            message=(
                event.content if isinstance(event, MessageEvent) else None
            ),
        )
        if isinstance(event, ToolCallEvent):
            context.execution.start_tools(event.calls)
        return projection

    def project(
        self,
        definition: "TaskDefinition",
        task: GeneratedTask,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentGenerationResult:
        metadata = dict(metadata) if metadata is not None else build_agent_metadata(definition)
        initial_attributes = deepcopy(
            task.requests[0].state_before.attributes
            if task.requests
            else definition.situation_init.attributes
        )
        context = HRProjectionContext(
            execution=TraceExecutionContext(
                attributes=initial_attributes,
                known_robots=tuple(
                    declared_known_robots(initial_attributes)
                ),
            ),
            history=[],
        )

        examples: list[dict[str, Any]] = []
        for request in task.requests:
            try:
                if (
                    context.execution.attributes
                    != request.state_before.attributes
                ):
                    raise RuntimeError(
                        "Trace attributes do not match request state_before: "
                        f"trace={context.execution.attributes!r}, "
                        f"state={request.state_before.attributes!r}."
                    )

                for event in request.perfect_trace.events:
                    projection = self._process_event(context, event)
                    if projection is None:
                        continue

                    example = self.action_projector.project(projection)
                    examples.append(example)
                    context.history.extend([
                        HRHistoryEntry(
                            author=projection.instruction.author,
                            content=projection.instruction.content,
                        ),
                        HRHistoryEntry(
                            author="MODEL",
                            content=json.dumps(
                                example["target"],
                                ensure_ascii=False,
                            ),
                        ),
                    ])
                    context.pending_instruction = None

                if (
                    context.execution.attributes
                    != request.state_after.attributes
                ):
                    raise RuntimeError(
                        "Trace attributes do not reproduce request state_after: "
                        f"trace={context.execution.attributes!r}, "
                        f"state={request.state_after.attributes!r}."
                    )
            except Exception as error:
                raise RuntimeError(
                    f"[Invalid Perfect Trace at {request.request_name}] {error}"
                ) from error

        if context.pending_instruction is not None:
            raise RuntimeError(
                "Perfect trace ended with an unconsumed instruction: "
                f"{context.pending_instruction.content!r}."
            )
        context.execution.validate_finished()

        return AgentGenerationResult(
            channels={"hr": examples},
            metadata=metadata,
        )

    def build_channel_configs(
        self,
        *,
        common_config: Mapping[str, Any],
        stats: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, dict[str, Any]]:
        return {
            "hr": {
                **common_config,
                "mode": "history_reactive",
                "export_kind": "history_reactive_action",
                "data_folder": ".",
                "tools": list(metadata.get("tools", [])),
                "persistent_rules": list(
                    metadata.get("persistent_rules", [])
                ),
                "stats": dict(stats),
            }
        }

    def render_projected(self, projected: AgentGenerationResult, options: ExportOptions) -> dict[str, list[dict[str, Any]]]:
        from ..rendering import Renderer
        renderer = Renderer()
        return {'commander': [renderer.render('commander', {
            **row['input'], 'tools': projected.metadata['tools'],
            'persistent_rules': projected.metadata['persistent_rules'], 'output': row['target'],
        }) for row in projected.channels['hr']]}

    @property
    def renderer(self):
        from ..rendering import Renderer
        return Renderer()
