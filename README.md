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
