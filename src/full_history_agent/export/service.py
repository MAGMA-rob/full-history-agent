from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
import random
import re
from typing import Any
from magma_core.protocol.agent_export import ExportRequest, ExportResponse, ExportResult, ExportRecord, ExportExample
from magma_core.protocol.agent_coaching import InternalStep


def replace_names(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        if not replacements:
            return value
        pattern = r"(?<![\w])(?:" + '|'.join(re.escape(name) for name in sorted(replacements, key=len, reverse=True)) + r")(?![\w])"
        return re.sub(pattern, lambda match: replacements[match.group()], value)
    if isinstance(value, list):
        return [replace_names(item, replacements) for item in value]
    if isinstance(value, dict):
        return {replace_names(key, replacements): replace_names(item, replacements) for key, item in value.items()}
    return value


class Exporter:
    def process(self, request: ExportRequest) -> ExportResponse:
        results = []
        for example in request.examples:
            try:
                if request.options:
                    raise ValueError('No private options are supported by this exporter')
                if example.candidate.response_status != 'completed':
                    results.append(ExportResult(example_id=example.example_id, status='skipped', reason='Invalid agent response'))
                    continue
                rows = self.build_rows(example)
                records = []
                for dataset, source in rows:
                    signature = sha256(json.dumps([dataset, source, example.input.attributes.get('known_robots', [])], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    robots = example.input.attributes.get('known_robots', [])
                    if not isinstance(robots, list) or any(not isinstance(robot, str) or not robot for robot in robots):
                        raise ValueError('known_robots must be a list of nonempty names')
                    for variant in range(request.num_variants):
                        rng = random.Random(f'{request.seed}:{signature}:{variant}')
                        names = [f'robot_{number}' for number in rng.sample(range(10000), len(robots))]
                        row = replace_names(deepcopy(source), dict(zip(robots, names)))
                        # Values remain consistent with the stored rendered prompt. Do not shuffle
                        # structured fields independently of the prompt used for this example.
                        encoded = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                   for key, value in row.items()}
                        record_id = sha256(f'{request.seed}:{signature}:{variant}'.encode()).hexdigest()
                        records.append(ExportRecord(record_id=record_id, dataset=dataset, data=encoded))
                results.append(ExportResult(example_id=example.example_id, status='exported' if records else 'skipped',
                                            records=records, reason=None if records else 'No usable internal model calls'))
            except (ValueError, TypeError, KeyError, IndexError) as error:
                results.append(ExportResult(example_id=example.example_id, status='error', reason=str(error)))
        return ExportResponse(request_id=request.request_id, results=results)

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
