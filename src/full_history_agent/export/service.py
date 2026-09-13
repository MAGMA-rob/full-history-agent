"""Local export of recorded executions; never initializes inference."""
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any
from magma_core.protocol.agent_export import ExportExample, ExportRecord, ExportResult, GenExporter
from magma_core.protocol.agent_coaching import InternalStep
from .. import __version__
from .rendering import Renderer


class Exporter(GenExporter):
    agent_id = 'full-history-agent'
    agent_version = __version__

    def __init__(self) -> None:
        self.renderer = Renderer()

    def export(self, examples: list[ExportExample], *, options: dict[str, Any] | None = None) -> list[ExportResult]:
        if options:
            raise ValueError('No private export options are supported')
        results = []
        for example in examples:
            try:
                records = []
                if example.candidate.response_status == 'completed':
                    for index, (channel, row) in enumerate(self.build_rows(example)):
                        data = self.renderer.render(channel, row)
                        identity = json.dumps([example.example_id, channel, index, data], sort_keys=True, ensure_ascii=False)
                        records.append(ExportRecord(record_id=sha256(identity.encode()).hexdigest(), dataset=channel, data=data))
                results.append(ExportResult(example_id=example.example_id,
                    status='exported' if records else 'skipped', records=records,
                    reason=None if records else 'No usable internal model calls'))
            except (ValueError, TypeError, KeyError, IndexError) as error:
                results.append(ExportResult(example_id=example.example_id, status='error', reason=str(error)))
        return results

    def build_rows(self, example: ExportExample) -> list[tuple[str, dict[str, Any]]]:
        rows = []
        memory = deepcopy(example.input.memory)
        for raw_step in example.candidate.internal_steps:
            step = InternalStep.model_validate(raw_step)
            if step.component != 'commander':
                continue
            decision = example.candidate.output
            actions = [{call.target_robot_name: {'name': call.name, 'arguments': call.arguments}}
                       for call in decision.tool_calls]
            action = {robot: value for item in actions for robot, value in item.items()}
            if len(action) != len(actions):
                action = actions
            elements = step.input_elements
            rows.append(('commander', {
                'tools': example.input.tools, 'persistent_rules': memory.get('memory_list', []),
                'memory': memory, 'attributes': example.input.attributes,
                'instruction': example.input.instruction.content,
                'instruction_role': 'USER' if example.input.instruction.type == 'user' else 'SYSTEM',
                'history': memory.get('history', []), 'summary': memory.get('summary', ''),
                'output': {'say': decision.say, 'action': action},
                'full_prompt': step.full_prompt, 'input_elements': elements, 'output_raw': step.output_raw,
            }))
        return rows
