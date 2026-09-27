# Assistant completions are serialized by the same code as coaching targets.
# Standard content/tool_calls are also supplied for compatible custom templates.
LLAMA_CHAT_TEMPLATE = """{{- bos_token }}
{%- for message in messages %}
{{- '<|start_header_id|>' + ('ipython' if message['role'] == 'tool' else message['role']) + '<|end_header_id|>\n\n' }}
{%- if message['role'] == 'system' %}
{{- message['content'] }}
{%- if tools %}
{{- '\n\nAvailable tools (JSON schemas):\n' }}{{- tools | tojson }}
{%- endif %}
{{- '<|eot_id|>' }}
{%- elif message['role'] == 'assistant' %}
{{- message['llama_completion'] }}
{%- else %}
{{- message['content'] + '<|eot_id|>' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
