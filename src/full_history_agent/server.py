"""Single-agent HTTP server; model imports are deferred until startup."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path

from magma_core.protocol.agent import AgentHealth, AgentInfo, AgentRequest, AgentResponse
from magma_core.protocol.agent_coaching import SpecializedCoachingRequest, SpecializedCoachingResponse
from . import __version__
from .config import Settings


def create_app(settings: Settings):
    from fastapi import FastAPI, HTTPException

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-history-agent-inference")
        app.state.executor = executor
        app.state.runtime = None
        app.state.coaching = None
        coaching_pool = None
        loop = asyncio.get_running_loop()
        try:
            from .runtime.agent import Runtime
            app.state.runtime = await loop.run_in_executor(executor, Runtime, settings)
            if settings.coaching_backends:
                from magma_core.workers import LMWorkerPool
                from .coaching.service import CoachingService
                coaching_pool = await asyncio.to_thread(LMWorkerPool, settings.coaching_backends)
                app.state.coaching = CoachingService(coaching_pool)
            yield
        finally:
            # An inference already running remains exclusive even after HTTP cancellation.
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            app.state.runtime = None
            if coaching_pool is not None:
                await asyncio.to_thread(coaching_pool.close)

    app = FastAPI(title="full-history-agent", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=AgentHealth)
    async def health():
        if getattr(app.state, "runtime", None) is None:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return AgentHealth()

    @app.get("/v1/info", response_model=AgentInfo)
    async def info():
        return AgentInfo(agent_id="full-history-agent", agent_version=__version__,
                         specialized_coaching=["failure", "suboptimal", "format"] if settings.coaching_backends else [],
                         coaching_resume=True,
                         capabilities={"inference": True, "coaching": bool(settings.coaching_backends), "export": False})

    @app.post("/v1/coaching", response_model=SpecializedCoachingResponse)
    async def coaching(request: SpecializedCoachingRequest):
        service = getattr(app.state, "coaching", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Specialized coaching is not configured")
        return await asyncio.to_thread(service.process, request)

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

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="full-history-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="JSON configuration file; explicit CLI values override it")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prompt-log-dir")
    models = ['model']
    for name in models:
        parser.add_argument("--" + name, dest=name + "_path")
        prefix = "" if name == "model" else name + "-"
        parser.add_argument("--" + ("model-format" if name == "model" else prefix + "format"), dest=name + "_format", choices=["auto", "magma", "qwen", "gpt-oss"])
        parser.add_argument("--" + prefix + "quantization", dest=name + "_quantization", choices=["auto", "4bit", "8bit", "none"])
        parser.add_argument("--" + prefix + "dtype", dest=name + "_dtype", choices=["auto", "float16", "bfloat16", "float32"])
        parser.add_argument("--" + prefix + "max-new-tokens", dest=name + "_max_new_tokens", type=int)
        for option in ("device_map", "gpu_memory_limit", "offload_folder", "attn_implementation", "chat_template", "output_style"):
            parser.add_argument("--" + prefix + option.replace("_", "-"), dest=name + "_" + option)
        for option in ("use_cache", "enable_thinking", "allow_cpu_offload"):
            parser.add_argument("--" + prefix + option.replace("_", "-"), dest=name + "_" + option,
                                action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        for name in ("host", "port", "prompt_log_dir", ""):
            if name and args.get(name) is not None:
                config[name] = args[name]
        for name in models:
            value = config.get(name, {})
            if isinstance(value, str):
                value = {"path": value}
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a path or configuration object")
            value = value.copy()
            for key, option in args.items():
                if key.startswith(name + "_") and option is not None:
                    value[key[len(name) + 1:]] = option
            config[name] = value
        settings = Settings.model_validate(config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    import uvicorn
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
