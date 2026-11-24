"""Tests for decoupled STT reading and streaming TTS in SonioxWorker."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from dubber.types import AudioBlock
from dubber.workers.soniox_worker import SonioxWorker


@pytest.mark.asyncio
async def test_soniox_worker_streams_pcm_chunks_immediately(tmp_path):
    """Verify SonioxWorker emits audio chunk upon first PCM fragment before TTS completes."""
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    tts_started = asyncio.Event()
    tts_finish = asyncio.Event()

    async def streaming_tts(text: str):
        tts_started.set()
        # Emit first chunk immediately
        yield np.ones(2400, dtype=np.int16).tobytes()
        # Wait for test to observe chunk before finishing
        await tts_finish.wait()
        # Emit final chunk
        yield np.ones(2400, dtype=np.int16).tobytes()

    mock_ws = AsyncMock()

    async def fake_recv():
        yield json.dumps({
            "tokens": [
                {
                    "text": "Streaming voice test",
                    "translation_status": "translation",
                    "is_final": True,
                }
            ]
        })
        while True:
            await asyncio.sleep(0.1)

    mock_ws.__aiter__.side_effect = fake_recv

    class FakeWsConnectContext:
        async def __aenter__(self):
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=FakeWsConnectContext()):
        worker = SonioxWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="test-key",
            tts_provider=streaming_tts,
        )
        await worker.start()

        pcm = np.zeros(320, dtype=np.int16).tobytes()
        await queue.put(AudioBlock("sess", 0, 0, 1.0, pcm))

        # First chunk must arrive before tts_finish is set!
        chunk1 = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk1 is not None
        assert len(chunk1.pcm_data) == 4800  # 2400 samples * 2 bytes

        # Now let TTS finish
        tts_finish.set()

        # Second chunk arrives
        chunk2 = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk2 is not None
        assert len(chunk2.pcm_data) == 4800

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_tts_queue_overflow_drops_stale_non_blocking(tmp_path):
    """Verify STT does not block when TTS queue is full, dropping stale items."""
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    tts_block_event = asyncio.Event()

    async def blocking_tts(text: str):
        await tts_block_event.wait()
        yield np.zeros(480, dtype=np.int16).tobytes()

    mock_ws = AsyncMock()

    async def fake_recv():
        for i in range(5):
            yield json.dumps({
                "tokens": [
                    {
                        "text": f"Phrase number {i}",
                        "translation_status": "translation",
                        "is_final": True,
                    }
                ]
            })
            await asyncio.sleep(0.01)
        while True:
            await asyncio.sleep(0.1)

    mock_ws.__aiter__.side_effect = fake_recv

    class FakeWsConnectContext:
        async def __aenter__(self):
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=FakeWsConnectContext()):
        worker = SonioxWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="test-key",
            tts_provider=blocking_tts,
            tts_queue_size=2,  # Tiny queue to test overflow
        )
        await worker.start()

        pcm = np.zeros(320, dtype=np.int16).tobytes()
        for i in range(5):
            await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

        # Give time for all 5 phrases to be received by STT
        await asyncio.sleep(0.1)

        # All 5 phrases MUST be logged even though TTS was blocked!
        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

        # Latest transcript in state is Phrase number 4
        state = worker.get_state()
        assert state.last_transcript == "Phrase number 4"

        # Overflow was flagged
        assert getattr(worker, "_tts_dropped_count", 0) > 0

        # Unblock TTS and cleanup
        tts_block_event.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_tts_failure_preserves_stt_transcripts(tmp_path):
    """Verify that when TTS fails on phrase 1, phrase 2 is still logged and updated."""
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    call_count = 0

    async def failing_first_tts(text: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("TTS connection error")
        yield np.ones(480, dtype=np.int16).tobytes()

    mock_ws = AsyncMock()

    async def fake_recv():
        yield json.dumps({
            "tokens": [
                {
                    "text": "Initial sentence",
                    "translation_status": "translation",
                    "is_final": True,
                }
            ]
        })
        await asyncio.sleep(0.05)
        yield json.dumps({
            "tokens": [
                {
                    "text": "Second sentence arrives",
                    "translation_status": "translation",
                    "is_final": True,
                }
            ]
        })
        while True:
            await asyncio.sleep(0.1)

    mock_ws.__aiter__.side_effect = fake_recv

    class FakeWsConnectContext:
        async def __aenter__(self):
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=FakeWsConnectContext()):
        worker = SonioxWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="test-key",
            tts_provider=failing_first_tts,
        )
        await worker.start()

        pcm = np.zeros(320, dtype=np.int16).tobytes()
        await queue.put(AudioBlock("sess", 0, 0, 1.0, pcm))
        await queue.put(AudioBlock("sess", 1, 320, 1.02, pcm))

        # Chunk from second sentence should eventually arrive
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None

        # Both sentences must be logged
        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert "Initial sentence" in lines[0]
        assert "Second sentence arrives" in lines[1]

        # Last transcript reflects second sentence
        state = worker.get_state()
        assert state.last_transcript == "Second sentence arrives"

        await worker.stop()
