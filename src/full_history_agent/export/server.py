"""Dedicated export HTTP process: imports no inference runtime or weights."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from magma_core.protocol.agent import AgentHealth, AgentInfo
from magma_core.protocol.agent_export import EXPORT_VERSION, ExportRequest, ExportResponse
from .. import __version__
from .service import Exporter


def create_app():
    from fastapi import FastAPI, HTTPException
    exporter = Exporter()
    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dataset-export')
        app.state.executor = executor
        try:
            yield
        finally:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
    app = FastAPI(lifespan=lifespan)
    @app.get('/health', response_model=AgentHealth)
    async def health():
        return AgentHealth()
    @app.get('/v1/info', response_model=AgentInfo)
    async def info():
        return AgentInfo(agent_id='full-history-agent', agent_version=__version__, export_version=EXPORT_VERSION,
                         capabilities={'inference': False, 'coaching': False, 'export': True})
    @app.post('/v1/export', response_model=ExportResponse)
    async def export(request: ExportRequest):
        if (request.agent_id, request.agent_version) != ('full-history-agent', __version__):
            raise HTTPException(422, 'Producer identity/version mismatch')
        return await asyncio.get_running_loop().run_in_executor(app.state.executor, exporter.process, request)
    return app


def main():
    parser = argparse.ArgumentParser(description='Run the model-free agent export service')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8100)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
