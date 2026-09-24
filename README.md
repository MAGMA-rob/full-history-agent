# full-history-agent

Reference agent implementation for MAGMA-GEN and MAGMA-BENCH, using the MAGMA
HTTP protocol v2. Install it from source and adapt the runtime to your own model.
Python 3.12 is required by the current MAGMA stack.

## Installation

In the same Python environment as MAGMA-GEN:

```bash
git clone https://github.com/MAGMA-rob/full-history-agent.git
cd full-history-agent
python -m pip install -e .
```

The editable installation uses your local code and makes the agent's dataset
exporter available to MAGMA-GEN. It works alongside MAGMA packages installed
from PyPI.

Recent Qwen models using linear attention can use optimized CUDA kernels:

```bash
python -m pip install -e '.[linear-attention]'
```

The extra installs `flash-linear-attention` and `causal-conv1d`. It is optional:
without it, Transformers falls back to correct but slower PyTorch implementations.
Keep the default installation for models that do not need these kernels.

## Model clients

The agent provides three model-format clients:

- `magma` handles models trained with the MAGMA conversation format.
- `qwen` handles Qwen2 and Qwen3 chat and tool-call formats. See
  `config.qwen.example.json`.
- `harmony` handles GPT-OSS through the Harmony protocol. See
  `config.harmony.example.json`.

The default `auto` format detects MAGMA chat templates, GPT-OSS models and Qwen
models. Set `model.format` explicitly when using a custom model or chat template.

## Configuration and launch

```bash
full-history-agent --model /path/to/model
```

For Qwen, start from the supplied configuration:

```bash
cp config.qwen.example.json config.qwen.json
full-history-agent --config config.qwen.json
```

For a local Qwen3.5 checkpoint, set `model.path` to its directory and
`model.quantization` to `"fp8"` to quantize the weights at load time. Keep
`model.dtype` as `"bfloat16"` for unquantized modules. The loader preserves the
vision tower, output head and small recurrent gate projections in their original
precision. The checkpoint on disk is unchanged.
With Transformers 5.17, use a CUDA GPU with compute capability 8.9 or newer;
on mixed-generation machines select a compatible GPU with `model.device_map`
(for example `"cuda:0"`) instead of `"auto"`. GPU indices refer to the devices
visible to the process, so account for `CUDA_VISIBLE_DEVICES` when launching.
The Docker image pins PyTorch 2.9.1 with Triton 3.5.1. Rebuild older images:
Triton 3.4 cannot import the autotuner used by the Transformers 5.17 FP8 kernel
and fails on the first inference request with a `JITFunction` import error.

For GPT-OSS, start from `config.harmony.example.json` instead.
The adapter exposes the internal `ask_user` and `execute_parallel` functions to
the model; they are converted to MAGMA responses and are never sent to the
robot tool executor as regular tools.

Command-line values override their JSON equivalents. For example:

```bash
full-history-agent --config config.qwen.json \
  --model /path/to/qwen \
  --port 9000 \
  --max-new-tokens 8192
```

For a local server, set `magma_agent_address` to `http://localhost:8888` in your
MAGMA configuration. Use `full-history-agent --help` for available options.

### Harmony history window

Set `model.history_max_messages` to a positive integer (for example `20`) to
include only the last N entries of `memory.history` in Harmony prompts. Omit it
or set it to `null` to keep the full history. The CLI override is
`--history-max-messages 20`. Zero, negative numbers, booleans, strings and
fractional values in JSON are rejected. A configured limit is supported only
for Harmony, including when the model format is detected automatically.

This counts historical entries, not tokens or rendered Harmony messages.
System/developer instructions, permanent rules, tools and the current input
with its attributes are always included outside that limit. The cutoff is
strict: a tool result whose call was removed is rendered as standalone context.
No summarization or preservation of the original task outside the window is added.

The returned memory remains complete, including GPT-OSS analysis indices.
Inference and coaching use the same window. Logged `input_elements.history`
contains the selected entries, `history_start_index` gives their original offset,
and `full_prompt` records the actual model prompt. This reduces prompt context,
not the size of returned memory or a guaranteed token budget.

## Dataset export

Run from the directory containing your generation output:

```bash
magma-gen export output --agent full-history-agent --skip-pre-made
```

Export runs locally in the MAGMA-GEN environment and does not require the agent
server or a loaded model.

## Documentation

See the [official documentation](https://magma-rob.github.io/docs/intro) for model
integration, configuration, and the HTTP protocol.

## License

[BSD 2-Clause](LICENSE).
