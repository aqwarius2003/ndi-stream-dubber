import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from dubber.types import AudioBlock, BroadcastAudioChunk
from dubber.workers.base import BaseWorker
from dubber.workers.soniox_worker import SonioxWorker


@pytest.mark.asyncio
async def test_soniox_worker_mock_flow(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")
    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.03,
        simulated_tts_delay_s=0.03,
    )
    await worker.start()

    pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
    for i in range(5):
        blk = AudioBlock("sess", i, i * 320, 10.0 + i * 0.02, pcm_20ms)
        await queue.put(blk)

    chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.5)
    assert chunk is not None
    assert isinstance(chunk, BroadcastAudioChunk)
    assert chunk.source_branch == "soniox"
    assert chunk.sample_rate == 48000

    # Check transcript was logged
    state = worker.get_state()
    assert state.status in ("READY", "RUNNING")
    assert state.last_audio_time > 0
    assert len(state.last_transcript) > 0

    with open(log_file, "r", encoding="utf-8") as f:  # noqa: ASYNC230
        lines = f.readlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["branch"] == "soniox"
        assert "text" in data

    await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_context_manager(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    async with SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.01,
        simulated_tts_delay_s=0.01,
    ) as worker:
        assert isinstance(worker, BaseWorker)
        state = worker.get_state()
        assert state.status in ("READY", "RUNNING")

        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            blk = AudioBlock("sess_ctx", i, i * 320, 1.0 + i * 0.02, pcm_20ms)
            await queue.put(blk)

        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None
        assert chunk.sample_rate == 48000

    assert worker.get_state().status == "STOPPED"


@pytest.mark.asyncio
async def test_soniox_worker_transcript_logged_independent_of_tts_failure(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    async def failing_tts(text: str) -> bytes:
        raise RuntimeError("TTS synthesis service crashed!")

    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.02,
        simulated_tts_delay_s=0.02,
        tts_provider=failing_tts,
    )
    await worker.start()

    pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
    for i in range(5):
        blk = AudioBlock("sess_fail", i, i * 320, 20.0 + i * 0.02, pcm_20ms)
        await queue.put(blk)

    # Wait long enough for STT delay and TTS attempt
    await asyncio.sleep(0.1)

    # Transcript MUST be logged despite TTS failure
    lines = Path(log_file).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    data = json.loads(lines[0])
    assert data["branch"] == "soniox"
    assert "text" in data

    state = worker.get_state()
    assert len(state.last_transcript) > 0
    # No audio chunk should have made it through successfully
    assert worker.get_state().status == "ERROR"

    await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_mock_loop_non_blocking(tmp_path):
    queue = asyncio.Queue(maxsize=50)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    # Delays of 0.2s each
    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.1,
        simulated_tts_delay_s=0.1,
        batch_size=2,
    )
    await worker.start()

    pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
    # Put 10 blocks (5 batches of 2)
    for i in range(10):
        await queue.put(AudioBlock("sess_nb", i, i * 320, 1.0 + i * 0.02, pcm_20ms))

    # Queue should be consumed quickly without waiting sequentially for STT+TTS delays
    await asyncio.sleep(0.05)
    assert queue.empty()

    # Wait for chunks to be emitted
    chunks = []
    for _ in range(5):
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        chunks.append(chunk)

    assert len(chunks) == 5
    await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_live_mode_mocked(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    sent_messages = []

    mock_ws = AsyncMock()

    async def fake_send(msg):
        sent_messages.append(msg)

    mock_ws.send = fake_send

    async def fake_recv():
        # First message: soniox streaming translated tokens
        yield json.dumps({
            "tokens": [
                {
                    "text": "Cardiovascular risk reduction",
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
            api_key="test-soniox-key",
            target_language="en",
        )
        await worker.start()

        # Send 5 blocks of 20ms audio
        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            blk = AudioBlock("sess_live", i, i * 320, 5.0 + i * 0.02, pcm_20ms)
            await queue.put(blk)

        # Wait for output chunk
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None
        assert chunk.sample_rate == 48000
        assert chunk.source_branch == "soniox"

        # Check initial config message sent to websocket
        assert len(sent_messages) >= 2  # config json + audio bytes
        config_msg = json.loads(sent_messages[0])
        assert config_msg["model"] == "stt-rt-v6"
        assert config_msg["translation"]["target_language"] == "en"

        # Check transcript and state
        state = worker.get_state()
        assert state.status == "RUNNING"
        assert state.last_transcript == "Cardiovascular risk reduction"

        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["branch"] == "soniox"
        assert data["text"] == "Cardiovascular risk reduction"

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_live_mode_tts_failure_preserves_transcript(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    mock_ws = AsyncMock()

    async def fake_recv():
        yield json.dumps({
            "tokens": [
                {
                    "text": "Anticoagulation therapy",
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

    async def failing_tts(text: str) -> bytes:
        raise RuntimeError("Live TTS error")

    with patch("websockets.connect", return_value=FakeWsConnectContext()):
        worker = SonioxWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="test-soniox-key",
            tts_provider=failing_tts,
        )
        await worker.start()

        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            blk = AudioBlock("sess_live_fail", i, i * 320, 5.0 + i * 0.02, pcm_20ms)
            await queue.put(blk)

        await asyncio.sleep(0.1)

        # Transcript MUST be logged despite TTS failure
        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["branch"] == "soniox"
        assert data["text"] == "Anticoagulation therapy"

        state = worker.get_state()
        assert state.last_transcript == "Anticoagulation therapy"

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_live_reconnect_on_error(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    call_count = 0
    mock_ws = AsyncMock()

    async def fake_recv():
        yield json.dumps({
            "tokens": [{"text": "Recovered speech", "translation_status": "translation"}]
        })
        while True:
            await asyncio.sleep(0.1)

    mock_ws.__aiter__.side_effect = fake_recv

    class FakeWsConnectContext:
        async def __aenter__(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Simulated websocket disconnect")
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=FakeWsConnectContext()):
        worker = SonioxWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="test-key",
            reconnect_delay_s=0.05,
        )
        await worker.start()

        # Send blocks while disconnected
        pcm = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(3):
            await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

        # Wait for reconnect delay to trigger retry and succeed
        await asyncio.sleep(0.15)

        # Send new blocks after reconnection
        for i in range(5):
            await queue.put(AudioBlock("sess", 10 + i, (10 + i) * 320, 2.0 + i * 0.02, pcm))

        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None
        assert chunk.source_branch == "soniox"

        state = worker.get_state()
        assert state.status == "RUNNING"
        assert state.last_transcript == "Recovered speech"

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_queue_overflow(tmp_path):
    queue = asyncio.Queue(maxsize=20)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.01,
        simulated_tts_delay_s=0.01,
        batch_size=2,
        queue_out_size=2,
    )
    await worker.start()

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    # Send 8 blocks -> 4 batches. Since queue_out_size=2, 2 chunks must be dropped
    for i in range(8):
        await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

    await asyncio.sleep(0.2)

    state = worker.get_state()
    assert state.dropped_blocks_count >= 1

    await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_stop_and_empty_queue(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
    )
    await worker.start()
    await worker.start()  # Idempotent start

    await worker.stop()
    await worker.stop()  # Idempotent stop

    assert worker.get_state().status == "STOPPED"
    chunk = await worker.get_output_chunk()
    assert chunk is None


@pytest.mark.asyncio
async def test_soniox_worker_live_non_final_tokens_not_sent_to_tts(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    tts_calls = []

    async def fake_tts(text: str) -> bytes:
        tts_calls.append(text)
        return (np.zeros(4800, dtype=np.int16)).tobytes()

    mock_ws = AsyncMock()

    async def fake_recv():
        # Step 1: Emit non-final token
        yield json.dumps({
            "tokens": [
                {
                    "text": "Partial clinical trial",
                    "translation_status": "translation",
                    "is_final": False,
                }
            ]
        })
        await asyncio.sleep(0.05)
        # Step 2: Emit finalized token
        yield json.dumps({
            "tokens": [
                {
                    "text": "results confirmed",
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
            tts_provider=fake_tts,
        )
        await worker.start()

        # Send blocks
        pcm = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

        # Wait for chunk from finalized token
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None

        # TTS must have been called ONLY with finalized token, NOT with non-final token
        assert "Partial clinical trial" not in tts_calls
        assert "results confirmed" in tts_calls

        # But transcript logging logged both/interim
        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        logged_texts = [json.loads(line)["text"] for line in lines]
        assert "Partial clinical trial" in logged_texts
        assert "results confirmed" in logged_texts

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_live_sent_batches_single_batch_pop(tmp_path):
    queue = asyncio.Queue(maxsize=50)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    mock_ws = AsyncMock()

    async def fake_recv():
        # Wait a short while so sender has pushed multiple batches to _sent_batches
        await asyncio.sleep(0.08)
        yield json.dumps({
            "tokens": [
                {
                    "text": "Sentence finished",
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
            batch_size=2,
        )
        await worker.start()

        # Send 10 blocks (5 batches) rapidly
        pcm = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(10):
            await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None

        # Verify only a single batch was popped per synthesized chunk (matching GeminiWorker)
        assert len(worker._sent_batches) == 4

        await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_mock_empty_tts_returns_early(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")

    # tts_provider returns empty bytes or None
    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.01,
        simulated_tts_delay_s=0.01,
        tts_provider=lambda text: b"",
    )
    await worker.start()

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    for i in range(5):
        await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

    # STT delay passes and transcript is logged
    await asyncio.sleep(0.05)
    lines = Path(log_file).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1

    # But output chunk was NOT emitted because empty TTS returns early
    assert worker._queue_out.empty()

    await worker.stop()


@pytest.mark.asyncio
async def test_soniox_worker_mock_audible_tone(tmp_path):
    """Verify SonioxWorker in mock mode emits an audible tone (~500 Hz, amplitude ~0.2*32767)."""
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_soniox.jsonl")
    worker = SonioxWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_stt_delay_s=0.01,
        simulated_tts_delay_s=0.01,
        batch_size=1,
    )
    await worker.start()
    try:
        pcm = (np.zeros(320, dtype=np.int16)).tobytes()
        await queue.put(AudioBlock("sess", 0, 0, 1.0, pcm))
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None
        samples = np.frombuffer(chunk.pcm_data, dtype=np.int16)
        assert len(samples) > 0
        max_amp = np.max(np.abs(samples))
        assert 5000 <= max_amp <= 7000

        from dubber.measure.latency import detect_speech_start

        onset = detect_speech_start(samples, sample_rate=48000, threshold=500.0)
        assert onset >= 0.0
    finally:
        await worker.stop()
