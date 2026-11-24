"""Loopback-only NDI operator console. The operator starts one live session."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.ingest.ndi_source import NDIAudioSource, discover_sources
from dubber.selector.selector import DubSelector
from dubber.sink.ndi_sink import NDIAudioSink

logger = logging.getLogger(__name__)
Direction = Literal["ru-en", "ru-de", "en-ru"]
OUTPUTS = {"ru-en": "Dub_EN", "ru-de": "Dub_DE", "en-ru": "Dub_RU"}


class SessionRequest(BaseModel):
    source: str = Field(min_length=1, max_length=1024)
    direction: Direction
    api_key: SecretStr | None = None


def reserved_source(name: str) -> bool:
    return any(name == out or name.endswith(f"({out})") for out in OUTPUTS.values())


def create_web_app(config: DubberConfig | None = None) -> FastAPI:
    settings = config or DubberConfig.from_env()
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            async with lock:
                if app.state.engine:
                    await app.state.engine.stop()

    app = FastAPI(title="PharmaLive Dubber", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.engine = None
    app.state.direction = None

    @app.middleware("http")
    async def local_controls(request: Request, call_next):
        if request.method == "POST" and request.headers.get("origin") not in (
            None, f"{request.url.scheme}://{request.headers['host']}"
        ):
            return PlainTextResponse("Local controls only", status_code=403)
        return await call_next(request)

    @app.get("/")
    async def page():
        return FileResponse(Path(__file__).with_name("console.html"))

    @app.get("/api/sources")
    async def sources():
        try:
            return [s for s in await asyncio.to_thread(discover_sources) if not reserved_source(s)]
        except (RuntimeError, OSError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/start")
    async def start(params: SessionRequest):
        async with lock:
            if app.state.engine:
                raise HTTPException(409, "Перевод уже запущен.")
            key = params.api_key.get_secret_value().strip() if params.api_key else settings.gemini_api_key
            if not key:
                raise HTTPException(400, "Введите ключ Gemini или задайте GEMINI_API_KEY в .env.")
            if reserved_source(params.source):
                raise HTTPException(400, "Выберите исходный сигнал vMix.")
            engine = DubberEngine(
                replace(settings, provider_mode="live", gemini_api_key=key),
                source=NDIAudioSource(params.source, session_id=uuid4().hex, exact_name=True),
                sink=NDIAudioSink(OUTPUTS[params.direction], realtime=True),
                selector=DubSelector(default_branch="gemini"),
                enable_soniox=False,
                target_language=params.direction.split("-")[1],
            )
            try:
                await engine.start()
            except (RuntimeError, OSError) as exc:
                raise HTTPException(503, str(exc).replace(key, "***")) from exc
            app.state.engine = engine
            app.state.direction = params.direction
            return status()

    @app.post("/api/stop")
    async def stop():
        async with lock:
            if app.state.engine:
                await app.state.engine.stop()
                app.state.engine = None
                app.state.direction = None
            return status()

    @app.get("/api/status")
    def status():
        engine = app.state.engine
        result = {"running": bool(engine), "key_configured": bool(settings.gemini_api_key)}
        if engine:
            now = time.monotonic()
            worker = engine.gemini_worker.get_state()
            error = engine.sink.error or engine.io_error or worker.error_message or ""
            result.update(
                direction=app.state.direction, source=engine.source.source_name,
                output=engine.sink.sink_name, provider=worker.status,
                input_active=now - engine.last_input_time < 2 and engine.last_input_time > 0,
                voice_active=now - engine.sink.last_voice_time < 2 and engine.sink.last_voice_time > 0,
                playout_dropped_samples=engine.sink.dropped_samples,
                buffer_target_ms=engine.sink.buffer.target_ms,
                local_silence_ms=engine.sink.buffer.local_silence_samples * 1000 // engine.sink.sample_rate,
                provider_zero_ms=engine.sink.buffer.provider_zero_samples * 1000 // engine.sink.sample_rate,
                transcript=worker.last_transcript,
                error=error.replace(engine.config.gemini_api_key, "***"),
            )
        return result

    return app
