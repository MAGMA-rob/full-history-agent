"""Single-agent HTTP server; model imports are deferred until startup."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path

from magma_core.protocol.agent import AgentHealth, AgentInfo, AgentRequest, AgentResponse
from . import __version__
from .config import Settings
from magma_core.workers.coaching_sessions import CoachingSessions, mount_coaching_routes


def create_app(settings: Settings):
    from fastapi import FastAPI, HTTPException

    sessions = CoachingSessions(supported=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-history-agent-inference")
        app.state.executor = executor
        app.state.runtime = None
        loop = asyncio.get_running_loop()
        try:
            from .runtime.agent import Runtime
            app.state.runtime = await loop.run_in_executor(executor, Runtime, settings)
            yield
        finally:
            # An inference already running remains exclusive even after HTTP cancellation.
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            app.state.runtime = None
            await asyncio.to_thread(sessions.close_all)

    app = FastAPI(title="full-history-agent", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=AgentHealth)
    async def health():
        if getattr(app.state, "runtime", None) is None:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return AgentHealth()

    @app.get("/v1/info", response_model=AgentInfo)
    async def info():
        return AgentInfo(agent_id="full-history-agent", agent_version=__version__,
                         specialized_coaching=["failure", "suboptimal", "format"],
                         coaching_session_version="1",
                         coaching_resume=True,
                         capabilities={"inference": True, "coaching": True})

    @app.post("/v1/responses", response_model=AgentResponse)
    async def responses(request: AgentRequest):
        runtime = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Models are not ready")
        try:
            runtime.validate_request(request)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result = await asyncio.get_running_loop().run_in_executor(app.state.executor, runtime.process, request)
        response = AgentResponse(result)
        response.validate_request(request)
        return response

    from .coaching.service import CoachingService
    mount_coaching_routes(app, sessions, CoachingService)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="full-history-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="JSON configuration file; explicit CLI values override it")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prompt-log-dir")
    parser.add_argument("--model", dest="model_path")
    parser.add_argument("--model-format", dest="model_format", choices=["auto", "magma", "qwen", "harmony"])
    parser.add_argument("--quantization", dest="model_quantization", choices=["auto", "4bit", "8bit", "fp8", "none"])
    parser.add_argument("--dtype", dest="model_dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--history-max-messages", dest="model_history_max_messages", type=int)
    parser.add_argument("--max-new-tokens", dest="model_max_new_tokens", type=int)
    parser.add_argument(
        "--reasoning-effort",
        dest="model_reasoning_effort",
        choices=["low", "medium", "high"],
    )
    for option in ("device_map", "gpu_memory_limit", "offload_folder", "attn_implementation", "chat_template", "output_style"):
        parser.add_argument("--" + option.replace("_", "-"), dest="model_" + option)
    for option in ("use_cache", "allow_cpu_offload"):
        parser.add_argument("--" + option.replace("_", "-"), dest="model_" + option,
                            action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        for name in ("host", "port", "prompt_log_dir"):
            if args.get(name) is not None:
                config[name] = args[name]
        model_config = config.get("model", {})
        if isinstance(model_config, str):
            model_config = {"path": model_config}
        if not isinstance(model_config, dict):
            raise ValueError("model must be a path or configuration object")
        model_config = model_config.copy()
        for key, option in args.items():
            if key.startswith("model_") and option is not None:
                model_config[key.removeprefix("model_")] = option
        config["model"] = model_config
        settings = Settings.model_validate(config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    import uvicorn
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
