# full-history-agent

Independent Python distribution implementing the MAGMA HTTP protocol v2.
It does not depend on another agent distribution. Python >=3.10 is required.

## Install and launch

From the workspace root:

```bash
python -m pip install -e ./magma-core-dev -e ./full-history-agent
full-history-agent --model /models/model
full-history-agent --config full-history-agent/config.example.json
python -m full_history_agent --help
```

The sample paths are placeholders. No model is downloaded by `--help` or by
importing configuration/server modules. Actual startup loads the configured
checkpoints. CLI values override JSON configuration; omitted fields retain
configuration values or documented defaults. A model in JSON can also be a path
string. One process exposes one agent; there are no architecture-selection flags.

Loads one LLM. Supports Qwen, GPT-OSS/Harmony and trained MAGMA templates.
Existing system prompts are retained, including their domain examples. A custom
chat template can be supplied with `--chat-template`.

Detection checks the checkpoint template for MAGMA variables before inspecting
its architecture, so a trained Qwen checkpoint is not mistaken for a base Qwen.
Use `--model-format magma|qwen|gpt-oss` only for ambiguous checkpoints.

`memory.history` is a list of `{author, content}` entries (`USER`, `SYSTEM`,
`MODEL`). `MODEL` content is a serialized `{say, action}` object. A valid response
appends the current instruction and decision. Invalid responses do not append
an exchange. `memory.memory_list` is a list of persistent rule strings.
Other memory keys are copied unchanged. Each candidate has independent memory.

Example: `full-history-agent --model /models/qwen --quantization 4bit`.
For GPT-OSS use `--model /models/gpt-oss --quantization auto`.

## Model settings

Each model accepts `path`, `format`, `quantization`, `dtype`, `max_new_tokens`,
`chat_template`, `output_style`, `device_map`, `gpu_memory_limit`,
`allow_cpu_offload`, `offload_folder`, `attn_implementation`, `use_cache`, and
`enable_thinking`. Format selects a tokenizer/output adapter, not an agent type.
Template/output-style options apply to trained adapters; thinking applies to Qwen.

Defaults: format/quantization/dtype `auto`, device map `auto`, attention `sdpa`,
cache enabled, Qwen thinking disabled, runtime CPU/GPU moves disabled. Quantization
`auto` retains native checkpoint quantization, otherwise uses 4 bits for the Qwen
adapter and no additional quantization for trained adapters. Explicit `4bit`,
`8bit`, `none` are supported for unquantized checkpoints. GPT-OSS requires `auto`;
re-quantizing an already quantized checkpoint is rejected. BitsAndBytes/CUDA
compatibility remains a deployment requirement.

Token limits default to 1,500 for Qwen commander, 600 for Qwen dispatcher, 1,024
for TSM/summarizer, and 2,500 for other commander/dispatcher adapters.
Use prefixed CLI options for multi-model agents, e.g. `--commander-quantization`
or `--dispatcher-max-new-tokens`. Server defaults: `0.0.0.0:8888`; optional
`--prompt-log-dir` writes raw prompt/response logs. Internal steps are always
returned even when disk logging is disabled.

## HTTP interface

- `POST /v1/responses`: complete input batch -> ordered list of final candidates.
- `GET /health`: `{"status":"ready"}` after model startup.
- `GET /v1/info`: protocol/agent versions and capabilities. Coaching/export are false.

The contract and its JSON Schemas live in `magma_core.protocol.agent`; see
`../magma-core-dev/docs/agent-protocol.md`. TLS can terminate at a reverse proxy.
One inference worker serializes whole requests while health/info stay accessible.
HTTP cancellation does not overlap another runtime with work still executing.

`extra_keys.inference_mode` is a boolean, default false: false samples using
existing adapter settings, true uses deterministic decoding. Different modes
in one batch are processed in separate model batches. Unknown extra keys are
preserved as opaque input and are not automatically interpreted as generation
arguments. For multiple outputs, deterministic decoding may yield identical
candidates.

`memory` in a result replaces the entire input memory. The server retains no
session memory. Successful output is exclusively `say` or `tool_calls`; tools
include `name`, `arguments`, and `target_robot_name`. A mixed model response is an
error even if the preserved prompt allowed it. An empty valid decision is allowed.

Each internal model call contains `id`, `component`, `kind="model_call"`, and
`data` with the actual rendered prompt, raw completion, parsing status and parsed
output. TSR adds its state snapshots and views. Errors retain these records.

## Migration and validation limits

This replaces the former multi-agent server. Gen/bench are intentionally not
migrated and cannot use protocol v2 until adapted. HTTP coaching and local dataset export are described below. Validation for this extraction covers compilation, imports and
packaging only; model inference and GPU quantization require separate validation.

## Coaching HTTP

Le runtime accepte `extra_keys.coaching` avec `{"kind":"replace_say","text":"..."}`
ou `{"kind":"replace_decision","decision":{"say":"...","tool_calls":[]}}`.
Il reconstruit l'historique depuis l'entrée fournie sans nouvelle inférence.
Les corrections invalides sont des erreurs de candidat.

`/v1/info` annonce `failure`, `suboptimal`, `format` et `coaching_session_version: "1"`.
La configuration de coaching vient exclusivement de `magma_gen` : supprimer l'ancien
champ `coaching_backends` du JSON de l'agent. Celui-ci conserve ses paramètres de modèle.

Avant les corrections, `magma_gen` enregistre les backends effectifs et le fournisseur
LLM ou humain via `PUT /v1/coaching/sessions/{run_id}`. Chaque requête à `/v1/coaching`
porte ce `run_id`. Une session est immuable et ses workers sont libérés par
`DELETE /v1/coaching/sessions/{run_id}`, après la fin des requêtes actives.

Les réponses contiennent `logs: [{"coaching_type": "failure", "content": "..."}]`,
y compris en cas d'abandon ou d'erreur. `magma_gen` écrit ces traces avec les diagnostics
locaux dans `output/<folder>/_coaching_logs/000001_failure.md`. Aucun montage du dossier
de sortie dans l'agent n'est nécessaire. Les propositions passent toujours par
`/v1/responses` avant leur exécution.

Les backends doivent être accessibles depuis l'agent : dans Docker, `localhost`
désigne le conteneur. Les adresses ne sont pas réécrites automatiquement.
Déployer ensemble les versions compatibles de `magma_core`, `magma_gen` et de l'agent.

## Local dataset export

```bash
magma-gen export output --agent full-history-agent --skip-pre-made
```

The installed package declares factories in `magma.export.gen` and
`magma.export.offpolicy`. Export is a local Python call; there is no export HTTP
server or export-server command. Runtime and coaching retain their HTTP APIs.
The gen adapter does not rename, paraphrase or shuffle recorded data, and loads
no inference model. Original prompts and raw responses remain in graph saves.

Install the `offpolicy` extra to use the demonstration projector. Both input
adapters use the agent's `export/rendering.py` for identical dataset columns and
JSON encoding. The output follows the off-policy JSON-array conventions, with
train/validation partitions by task and a versioned, verified export manifest.
Off-policy paraphrase and summary backends are initialized only when needed.
