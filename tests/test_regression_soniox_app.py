"""Regression tests for full application assembly with Soniox STT + streaming TTS.

Tests both core scenarios:
1. Early audio output: TTS server yields recognizable PCM fragment and delays stream
   completion. The fragment reaches the audio sink BEFORE TTS synthesis finishes.
2. Voice error isolation: TTS fails/hangs on first phrase, STT delivers second phrase.
   Second phrase appears in log and last_transcript, sink continues clocking 20ms blocks,
   /status shows voice error, and active route remains 'soniox'.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import websockets
from fastapi.testclient import TestClient

from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.ingest.base import AudioSource
from dubber.operator.server import create_app
from dubber.sink.base import AudioSink
from dubber.types import AudioBlock, BroadcastAudioChunk
from dubber.workers.gemini_worker import GeminiWorker


class ListAudioSink(AudioSink):
    """AudioSink capturing written chunks into a list for inspection."""

    def __init__(self) -> None:
        self.chunks: list[BroadcastAudioChunk] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        self.chunks.append(chunk)


class PushAudioSource(AudioSource):
    """AudioSource allowing manual push of AudioBlocks into the ingest loop."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AudioBlock] = asyncio.Queue()
        self._running = False

    def is_active(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def read_block(self) -> AudioBlock | None:
        if not self._running:
            return None
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.05)
        except TimeoutError:
            return None

    async def push_block(self, block: AudioBlock) -> None:
        await self.queue.put(block)


@pytest.mark.asyncio
async def test_regression_soniox_early_pcm_delivery_before_tts_completion(tmp_path):
    """Scenario 1: Recognizable PCM fragment reaches sink BEFORE TTS finishes."""
    tts_stream_finished = asyncio.Event()
    hold_completion = asyncio.Event()

    # Recognizable PCM payload: 2400 samples (50ms at 48kHz)
    recognizable_pcm = b"\xaa\x55" * 2400

    async def mock_stt_server(websocket: Any) -> None:
        first_audio = False
        async for msg in websocket:
            if isinstance(msg, bytes) and not first_audio:
                first_audio = True
                await websocket.send(
                    json.dumps({
                        "tokens": [
                            {
                                "text": "Cardiovascular risk reduction",
                                "translation_status": "translation",
                                "is_final": True,
                            }
                        ]
                    })
                )

    async def mock_tts_server(websocket: Any) -> None:
        cfg_msg = await websocket.recv()
        cfg = json.loads(cfg_msg)
        stream_id = cfg["stream_id"]

        # Text frame
        await websocket.recv()

        # Send recognizable audio chunk immediately (audio_end=False)
        await websocket.send(
            json.dumps({
                "stream_id": stream_id,
                "audio": base64.b64encode(recognizable_pcm).decode("ascii"),
                "audio_end": False,
            })
        )

        # Hold completion until test signals
        await hold_completion.wait()

        # Final audio_end frame
        await websocket.send(
            json.dumps({
                "stream_id": stream_id,
                "audio": "",
                "audio_end": True,
            })
        )
        tts_stream_finished.set()

    stt_srv = await websockets.serve(mock_stt_server, "127.0.0.1", 0)
    tts_srv = await websockets.serve(mock_tts_server, "127.0.0.1", 0)
    stt_port = stt_srv.sockets[0].getsockname()[1]
    tts_port = tts_srv.sockets[0].getsockname()[1]

    log_path = str(tmp_path / "transcript_soniox.jsonl")
    state_path = str(tmp_path / "state.json")

    config = DubberConfig(
        provider_mode="live",
        crossfade_ms=0,
        state_file_path=state_path,
        soniox_tts_endpoint_url=f"ws://127.0.0.1:{tts_port}",
    )

    source = PushAudioSource()
    sink = ListAudioSink()
    mock_gemini = GeminiWorker(queue_in=asyncio.Queue(), mode="mock")

    engine = DubberEngine(config, source=source, sink=sink, gemini_worker=mock_gemini)
    engine.soniox_worker.endpoint_url = f"ws://127.0.0.1:{stt_port}"
    engine.soniox_worker.transcript_path = log_path

    # Force switch route to soniox
    engine.selector.switch_branch("soniox", force=True)

    await engine.start()
    try:
        # Push 5 audio blocks to trigger STT batch send
        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            await source.push_block(AudioBlock("sess1", i, i * 320, 1.0 + i * 0.02, pcm_20ms))

        # Wait until recognizable PCM arrives at the audio sink
        start_wait = time.monotonic()
        arrived = False
        while time.monotonic() - start_wait < 3.0:
            for ch in sink.chunks:
                if recognizable_pcm in ch.pcm_data:
                    arrived = True
                    break
            if arrived:
                break
            await asyncio.sleep(0.02)

        # ASSERTION 1: The recognizable fragment reached the sink!
        assert arrived is True, "Recognizable PCM fragment failed to reach the audio sink"

        # ASSERTION 2: TTS stream completion was still being held when chunk arrived!
        assert not tts_stream_finished.is_set(), (
            "PCM chunk must reach output sink BEFORE TTS finishes synthesis"
        )

        # Now release the completion hold
        hold_completion.set()
        await asyncio.wait_for(tts_stream_finished.wait(), timeout=1.0)
        assert tts_stream_finished.is_set()

    finally:
        hold_completion.set()
        await engine.stop()
        stt_srv.close()
        tts_srv.close()
        await stt_srv.wait_closed()
        await tts_srv.wait_closed()


@pytest.mark.asyncio
async def test_regression_soniox_tts_failure_isolation_and_route_preservation(tmp_path):
    """Scenario 2: TTS fails on phrase 1, STT delivers phrase 2.
    Phrase 2 appears in transcript, sink clocks 20ms silence, /status shows voice error,
    route remains 'soniox'.
    """
    async def mock_stt_server(websocket: Any) -> None:
        async def send_phrases() -> None:
            await asyncio.sleep(0.05)
            await websocket.send(
                json.dumps({
                    "tokens": [
                        {
                            "text": "First phrase with broken audio",
                            "translation_status": "translation",
                            "is_final": True,
                        }
                    ]
                })
            )
            await asyncio.sleep(0.2)
            await websocket.send(
                json.dumps({
                    "tokens": [
                        {
                            "text": "Second phrase accurately transcribed",
                            "translation_status": "translation",
                            "is_final": True,
                        }
                    ]
                })
            )

        send_task = asyncio.create_task(send_phrases())
        try:
            async for _ in websocket:
                pass
        finally:
            send_task.cancel()

    async def mock_tts_server(websocket: Any) -> None:
        await websocket.recv()  # config frame
        await websocket.recv()  # text frame

        # Send TTS error frame
        await websocket.send(
            json.dumps({
                "error_type": "synthesis_failure",
                "error_code": 500,
                "error_message": "Audio synthesizer internal error",
            })
        )

    stt_srv = await websockets.serve(mock_stt_server, "127.0.0.1", 0)
    tts_srv = await websockets.serve(mock_tts_server, "127.0.0.1", 0)
    stt_port = stt_srv.sockets[0].getsockname()[1]
    tts_port = tts_srv.sockets[0].getsockname()[1]

    log_path = str(tmp_path / "transcript_soniox.jsonl")
    state_path = str(tmp_path / "state.json")

    config = DubberConfig(
        provider_mode="live",
        crossfade_ms=0,
        state_file_path=state_path,
        soniox_tts_endpoint_url=f"ws://127.0.0.1:{tts_port}",
    )

    source = PushAudioSource()
    sink = ListAudioSink()
    mock_gemini = GeminiWorker(queue_in=asyncio.Queue(), mode="mock")

    engine = DubberEngine(config, source=source, sink=sink, gemini_worker=mock_gemini)
    engine.soniox_worker.endpoint_url = f"ws://127.0.0.1:{stt_port}"
    engine.soniox_worker.transcript_path = log_path

    # Force switch route to soniox
    engine.selector.switch_branch("soniox", force=True)

    app = create_app(engine)
    client = TestClient(app)

    await engine.start()
    try:
        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()

        # 1. Push batch for phrase 1 (5 blocks)
        for i in range(5):
            await source.push_block(AudioBlock("sess2", i, i * 320, 1.0 + i * 0.02, pcm_20ms))

        # Wait for TTS error to register
        await asyncio.sleep(0.2)

        # 2. Push batch for phrase 2 (5 blocks)
        for i in range(5):
            await source.push_block(AudioBlock("sess2", 5 + i, (5 + i) * 320, 2.0 + i * 0.02, pcm_20ms))

        # Wait for phrase 2 to be received and processed
        await asyncio.sleep(0.3)

        # ASSERTION 1: Second phrase appears in transcript log
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 2
        texts = [json.loads(line)["text"] for line in lines]
        assert "Second phrase accurately transcribed" in texts

        # ASSERTION 2: last_transcript reflects second phrase
        state = engine.soniox_worker.get_state()
        assert state.last_transcript == "Second phrase accurately transcribed"

        # ASSERTION 3: Sink continues receiving 20-ms audio blocks (from silence heartbeat)
        assert len(sink.chunks) > 0
        silence_blocks = [
            c for c in sink.chunks
            if c.session_id in ("silence", "") and len(c.pcm_data) == 1920
        ]
        assert len(silence_blocks) > 0, "Silence heartbeat must clock 20ms blocks during voice failure"

        # ASSERTION 4: Operator API /status shows alert and voice error, and route remains 'soniox'
        res = client.get("/status")
        assert res.status_code == 200
        data = res.json()
        assert data["active_branch"] == "soniox", "Route MUST remain soniox (zero auto-fallback)"
        assert data["alert"] is True, "Alert MUST be active during voice failure"
        soniox_data = data["branches"]["soniox"]
        assert soniox_data["error_message"] is not None
        assert "TTS" in soniox_data["error_message"]

    finally:
        await engine.stop()
        stt_srv.close()
        tts_srv.close()
        await stt_srv.wait_closed()
        await tts_srv.wait_closed()
