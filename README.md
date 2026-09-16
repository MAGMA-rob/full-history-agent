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

## Launch

```bash
full-history-agent --model /path/to/model
```

Alternatively, edit `config.example.json` with your model settings and run:

```bash
full-history-agent --config config.example.json
```

For GPT-OSS with the Harmony adapter, use `config.harmony.example.json` instead.
The adapter exposes the internal `ask_user` and `execute_parallel` functions to
the model; they are converted to MAGMA responses and are never sent to the
robot tool executor as regular tools.

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
