from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from btc_intelligence.api.routes import router
from btc_intelligence.config import settings
from btc_intelligence.service import AppRuntime
from btc_intelligence.utils.logger import setup_logging


setup_logging(settings.log_level, settings.app_log_path)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router)


@app.on_event('startup')
async def on_startup() -> None:
    app.state.runtime = AppRuntime()
    await app.state.runtime.start()
    logger.info('App startup complete')


@app.on_event('shutdown')
async def on_shutdown() -> None:
    rt: AppRuntime = app.state.runtime
    await rt.stop()
    logger.info('App shutdown complete')


if __name__ == '__main__':
    uvicorn.run('btc_intelligence.main:app', host=settings.host, port=settings.port, reload=False)
