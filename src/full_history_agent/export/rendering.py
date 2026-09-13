"""Single dataset schema shared by recorded and demonstrated examples."""
import json
from typing import Any
from magma_core.protocol.agent_export import DatasetRenderer


class Renderer(DatasetRenderer):
    filenames = {'commander': 'hr_data.json'}

    def render(self, channel: str, example: dict[str, Any]) -> dict[str, Any]:
        if channel not in self.filenames:
            raise ValueError(f'Unsupported dataset channel: {channel}')
        row = {key: json.dumps(example.get(key, default), ensure_ascii=False)
               for key, default in (('tools', []), ('persistent_rules', []), ('memory', {}),
                                    ('attributes', {}), ('history', []))}
        row.update(instruction=example.get('instruction', ''),
                   instruction_role=example.get('instruction_role', 'USER'),
                   output=json.dumps(example['output'], ensure_ascii=False))

        return row
