import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dubber.types import AudioBlock, BroadcastAudioChunk
from dubber.workers.base import BaseWorker
from dubber.workers.gemini_worker import GeminiWorker


@pytest.mark.asyncio
async def test_gemini_worker_mock_flow(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.05,
    )
    await worker.start()

    # Send 5 blocks of 20ms audio (100 ms total)
    pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
    for i in range(5):
        blk = AudioBlock("sess", i, i * 320, 10.0 + i * 0.02, pcm_20ms)
        await queue.put(blk)

    # Read output chunk
    chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
    assert chunk is not None
    assert isinstance(chunk, BroadcastAudioChunk)
    assert chunk.source_branch == "gemini"
    assert chunk.sample_rate == 48000

    # Check transcript was logged
    state = worker.get_state()
    assert state.status in ("READY", "RUNNING")
    assert state.last_audio_time > 0
    assert len(state.last_transcript) > 0

    lines = Path(log_file).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    data = json.loads(lines[0])
    assert data["branch"] == "gemini"
    assert "text" in data

    await worker.stop()


@pytest.mark.asyncio
async def test_gemini_worker_context_manager(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")

    async with GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.01,
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
async def test_gemini_worker_live_mode_mocked(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")

    # Prepare fake 24 kHz response audio (240 samples = 10 ms at 24 kHz)
    pcm_24k = (np.ones(240, dtype=np.int16) * 1000).tobytes()

    fake_response = MagicMock()
    fake_response.server_content.input_transcription.text = "Здравствуйте"
    fake_response.server_content.output_transcription.text = "Hello world"
    fake_part = MagicMock()
    fake_part.inline_data.data = pcm_24k
    fake_response.server_content.model_turn.parts = [fake_part]

    async def fake_receive():
        yield fake_response
        # Keep waiting until cancelled or stopped
        while True:
            await asyncio.sleep(0.1)

    mock_session = AsyncMock()
    mock_session.receive = fake_receive
    mock_session.send_realtime_input = AsyncMock()

    class FakeLiveConnectContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client = MagicMock()
    mock_client.aio.live.connect.return_value = FakeLiveConnectContext()

    with patch("google.genai.Client", return_value=mock_client):
        worker = GeminiWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="fake-test-key",
            target_language="en",
        )
        await worker.start()

        # Send 5 blocks of 20ms audio (100 ms total)
        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        for i in range(5):
            blk = AudioBlock("sess_live", i, i * 320, 5.0 + i * 0.02, pcm_20ms)
            await queue.put(blk)

        # Wait for output chunk resampled to 48 kHz
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
        assert chunk is not None
        assert chunk.sample_rate == 48000
        assert chunk.source_branch == "gemini"
        # 240 samples at 24 kHz resampled by 2/1 -> 480 samples = 960 bytes
        assert len(chunk.pcm_data) == 480 * 2

        # Check call to send_realtime_input
        assert mock_session.send_realtime_input.called

        # Check state and transcript
        state = worker.get_state()
        assert state.status == "RUNNING"
        assert state.last_transcript == "Hello world"

        await worker.stop()
        assert worker.get_state().status == "STOPPED"


@pytest.mark.asyncio
async def test_gemini_worker_resilient_error_handling(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")

    # Simulate connection failure in live mode
    mock_client = MagicMock()
    mock_client.aio.live.connect.side_effect = ConnectionError("Simulated connection error")

    with patch("google.genai.Client", return_value=mock_client):
        worker = GeminiWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="fake-test-key",
            reconnect_delay_s=0.05,
        )
        await worker.start()

        # Wait shortly for connection attempt to fail
        await asyncio.sleep(0.1)

        state = worker.get_state()
        assert state.status == "ERROR"
        assert state.error_message is not None
        assert "Simulated connection error" in state.error_message

        # Verify worker did not crash and stop works cleanly
        await worker.stop()
        assert worker.get_state().status == "STOPPED"


@pytest.mark.asyncio
async def test_gemini_worker_stop_and_empty_queue(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
    )
    await worker.start()
    await worker.start()  # Calling start twice should be a no-op

    await worker.stop()
    await worker.stop()  # Calling stop twice should be safe

    assert worker.get_state().status == "STOPPED"
    chunk = await worker.get_output_chunk()
    assert chunk is None


@pytest.mark.asyncio
async def test_gemini_worker_queue_overflow(tmp_path):
    queue = asyncio.Queue(maxsize=20)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    # Tiny output queue of size 2
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.01,
        batch_size=2,
        queue_out_size=2,
    )
    await worker.start()

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    # Send 8 blocks -> 4 batches. Since queue_out_size=2, 2 chunks must be dropped
    for i in range(8):
        await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

    # Give time to process all batches
    await asyncio.sleep(0.2)

    state = worker.get_state()
    assert state.dropped_blocks_count >= 1

    await worker.stop()


@pytest.mark.asyncio
async def test_gemini_worker_reconnect_success(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")

    call_count = 0
    mock_session = AsyncMock()

    async def fake_receive():
        fake_response = MagicMock()
        fake_response.server_content.output_transcription.text = "Recovered transcript"
        fake_part = MagicMock()
        fake_part.inline_data.data = (np.zeros(240, dtype=np.int16)).tobytes()
        fake_response.server_content.model_turn.parts = [fake_part]
        yield fake_response
        while True:
            await asyncio.sleep(0.1)

    mock_session.receive = fake_receive
    mock_session.send_realtime_input = AsyncMock()

    class FakeLiveConnectContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def connect_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionResetError("First connection failed")
        return FakeLiveConnectContext()

    mock_client = MagicMock()
    mock_client.aio.live.connect.side_effect = connect_side_effect

    with patch("google.genai.Client", return_value=mock_client):
        worker = GeminiWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            reconnect_delay_s=0.05,
            batch_size=1,
        )
        await worker.start()

        # Send block so queue gets processed
        pcm_20ms = (np.zeros(320, dtype=np.int16)).tobytes()
        await queue.put(AudioBlock("sess", 0, 0, 0.0, pcm_20ms))

        # Wait for reconnection and output
        chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=2.0)
        assert chunk is not None
        assert worker.get_state().status == "RUNNING"
        assert worker.get_state().last_transcript == "Recovered transcript"
        assert call_count >= 2

        await worker.stop()


@pytest.mark.asyncio
async def test_gemini_worker_restart_no_leftover_none(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.01,
    )
    # First cycle
    await worker.start()
    await worker.stop()
    assert worker.get_state().status == "STOPPED"

    # Second cycle (restart)
    await worker.start()
    assert worker.get_state().status in ("READY", "RUNNING")

    # Send audio block to restarted worker
    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    for i in range(5):
        await queue.put(AudioBlock("sess_restart", i, i * 320, 1.0 + i * 0.02, pcm))

    chunk = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
    assert chunk is not None
    assert isinstance(chunk, BroadcastAudioChunk)
    assert chunk.sample_rate == 48000

    await worker.stop()


@pytest.mark.asyncio
async def test_gemini_worker_live_subtasks_cleanup_on_stop(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")

    async def fake_receive():
        while True:
            await asyncio.sleep(0.5)
            yield MagicMock()

    mock_session = AsyncMock()
    mock_session.receive = fake_receive
    mock_session.send_realtime_input = AsyncMock()

    class FakeLiveConnectContext:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client = MagicMock()
    mock_client.aio.live.connect.return_value = FakeLiveConnectContext()

    with patch("google.genai.Client", return_value=mock_client):
        worker = GeminiWorker(
            queue_in=queue,
            mode="live",
            transcript_path=log_file,
            api_key="fake-test-key",
        )
        spawned_tasks = []
        original_create_task = asyncio.create_task

        def spy_create_task(coro, *args, **kwargs):
            task = original_create_task(coro, *args, **kwargs)
            if kwargs.get("name") in ("gemini_live_sender", "gemini_live_receiver"):
                spawned_tasks.append(task)
            return task

        with patch("asyncio.create_task", side_effect=spy_create_task):
            await worker.start()

            # Allow live loop to connect and spawn sender/receiver subtasks
            await asyncio.sleep(0.05)

            assert len(spawned_tasks) == 2
            assert all(not t.done() for t in spawned_tasks)

            # Stop worker, cancelling _live_loop which must cancel and await subtasks
            await worker.stop()

            # Both subtasks must now be cancelled and done
            assert len(spawned_tasks) == 2
            for t in spawned_tasks:
                assert t.done()
                assert t.cancelled()


@pytest.mark.asyncio
async def test_gemini_worker_mock_continuous_consumption(tmp_path):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.1,
        batch_size=2,
    )
    await worker.start()

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    # Feed 4 blocks (2 batches)
    for i in range(4):
        await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))

    # Give very short time for mock loop to consume blocks from queue_in
    await asyncio.sleep(0.02)
    # Queue should be consumed immediately even though simulated delay (0.1s) is pending
    assert queue.empty()

    # After simulated delay, chunks arrive
    chunk1 = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
    chunk2 = await asyncio.wait_for(worker.get_output_chunk(), timeout=1.0)
    assert chunk1 is not None and chunk2 is not None

    await worker.stop()


@pytest.mark.asyncio
async def test_gemini_worker_dropped_block_logging(tmp_path, caplog):
    queue = asyncio.Queue(maxsize=10)
    log_file = str(tmp_path / "transcript_gemini.jsonl")
    worker = GeminiWorker(
        queue_in=queue,
        mode="mock",
        transcript_path=log_file,
        simulated_delay_s=0.01,
        batch_size=1,
        queue_out_size=1,
    )
    await worker.start()

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()
    with caplog.at_level(logging.WARNING):
        for i in range(3):
            await queue.put(AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, pcm))
        await asyncio.sleep(0.1)

    assert "GeminiWorker discarded block" in caplog.text
    await worker.stop()



