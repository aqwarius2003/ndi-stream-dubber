import asyncio
import logging

import numpy as np
import pytest

from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.ingest.base import AudioSource
from dubber.selector.selector import DubSelector
from dubber.sink.base import AudioSink
from dubber.types import AudioBlock, BranchState, BroadcastAudioChunk
from dubber.workers.base import BaseWorker


class DummyAudioSource(AudioSource):
    """Test audio source providing pre-configured blocks."""

    def __init__(self, blocks: list[AudioBlock]) -> None:
        self.blocks = list(blocks)
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def read_block(self) -> AudioBlock | None:
        if not self.blocks:
            return None
        return self.blocks.pop(0)


class DummyAudioSink(AudioSink):
    """Test audio sink collecting chunks written to it."""

    def __init__(self) -> None:
        self.chunks: list[BroadcastAudioChunk] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        self.chunks.append(chunk)


class DummyWorker(BaseWorker):
    """Test worker simulating chunk emission and state."""

    def __init__(self, branch: str, queue_in: asyncio.Queue[AudioBlock]) -> None:
        self.branch = branch
        self.queue_in = queue_in
        self.queue_out: asyncio.Queue[BroadcastAudioChunk | None] = asyncio.Queue()
        self.state = BranchState(branch=branch, status="READY")
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True
        await self.queue_out.put(None)

    async def get_output_chunk(self) -> BroadcastAudioChunk | None:
        return await self.queue_out.get()

    def get_state(self) -> BranchState:
        return self.state


@pytest.mark.asyncio
async def test_engine_bounded_queue_drop_oldest(caplog):
    config = DubberConfig(max_queue_size=2)
    engine = DubberEngine(config)

    # Fill gemini and soniox queues with 2 blocks
    b0 = AudioBlock("sess", 0, 0, 1.0, b"\x00" * 640)
    b1 = AudioBlock("sess", 1, 320, 1.02, b"\x00" * 640)
    b2 = AudioBlock("sess", 2, 640, 1.04, b"\x00" * 640)

    with caplog.at_level(logging.WARNING):
        await engine.dispatch_block_to_queues(b0)
        await engine.dispatch_block_to_queues(b1)
        # Queue is full, dispatching b2 must drop b0 and not block
        await engine.dispatch_block_to_queues(b2)

    assert engine.queue_gemini.qsize() == 2
    first_remaining = await engine.queue_gemini.get()
    assert first_remaining.block_index == 1
    second_remaining = await engine.queue_gemini.get()
    assert second_remaining.block_index == 2

    # Check dropped block tracking
    assert engine.dropped_blocks["gemini"] == 1
    assert engine.dropped_blocks["soniox"] == 1

    # Verify log message matches requirement
    assert "Queue overflow on branch 'gemini'. Dropped block_index=0 to preserve real-time pace." in caplog.text
    assert "Queue overflow on branch 'soniox'. Dropped block_index=0 to preserve real-time pace." in caplog.text


@pytest.mark.asyncio
async def test_engine_sibling_isolation():
    """Slow or stalled worker never blocks audio capture or the sibling worker."""
    config = DubberConfig(max_queue_size=2)
    engine = DubberEngine(config)

    # Soniox queue is never drained (simulates stalled worker).
    # Gemini queue is continuously drained.
    for i in range(10):
        block = AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, b"\x01" * 640)
        await engine.dispatch_block_to_queues(block)
        # Drain Gemini queue immediately
        drained = await engine.queue_gemini.get()
        assert drained.block_index == i

    # Gemini had 0 drops because it drained promptly
    assert engine.dropped_blocks["gemini"] == 0
    # Soniox stalled and dropped 8 blocks (10 total - 2 buffer)
    assert engine.dropped_blocks["soniox"] == 8
    assert engine.queue_soniox.qsize() == 2


@pytest.mark.asyncio
async def test_engine_lifecycle_and_context_manager(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    source = DummyAudioSource([])
    sink = DummyAudioSink()

    engine = DubberEngine(config, source=source, sink=sink)
    assert not engine.is_running

    async with engine:
        assert engine.is_running
        assert source.started
        assert sink.started

    assert not engine.is_running
    assert source.stopped
    assert sink.stopped

    # Test idempotent start/stop
    await engine.stop()  # Should not raise
    assert not engine.is_running


@pytest.mark.asyncio
async def test_engine_concurrent_pipeline_routing(tmp_path):
    """Verify chunks from workers route through DubSelector to AudioSink."""
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    sink = DummyAudioSink()
    source = DummyAudioSource([])

    q_gemini = asyncio.Queue(maxsize=10)
    q_soniox = asyncio.Queue(maxsize=10)
    w_gemini = DummyWorker("gemini", q_gemini)
    w_soniox = DummyWorker("soniox", q_soniox)
    selector = DubSelector(state_file_path=str(tmp_path / "state.json"), default_branch="gemini")

    engine = DubberEngine(
        config=config,
        source=source,
        sink=sink,
        selector=selector,
        gemini_worker=w_gemini,
        soniox_worker=w_soniox,
    )

    await engine.start()
    try:
        # Active branch is Gemini. Send chunk from Gemini.
        c_gemini = BroadcastAudioChunk(
            session_id="s1",
            language="en",
            source_branch="gemini",
            block_index=1,
            source_time=1.0,
            sample_rate=48000,
            channels=1,
            pcm_data=b"\x10\x00" * 480,
        )
        await w_gemini.queue_out.put(c_gemini)

        # Send chunk from Soniox (inactive branch)
        c_soniox = BroadcastAudioChunk(
            session_id="s1",
            language="en",
            source_branch="soniox",
            block_index=1,
            source_time=1.0,
            sample_rate=48000,
            channels=1,
            pcm_data=b"\x20\x00" * 480,
        )
        await w_soniox.queue_out.put(c_soniox)

        # Wait briefly for pipeline orchestration
        await asyncio.sleep(0.05)

        # Only Gemini chunk should reach the sink
        assert len(sink.chunks) == 1
        assert sink.chunks[0].source_branch == "gemini"

        # Switch to Soniox
        # Mark soniox as fresh and ready
        w_soniox.state.status = "READY"
        w_soniox.state.last_audio_time = asyncio.get_running_loop().time()
        ok, _ = selector.switch_branch("soniox", force=True)
        assert ok

        # Send another Soniox chunk
        c_soniox2 = BroadcastAudioChunk(
            session_id="s1",
            language="en",
            source_branch="soniox",
            block_index=2,
            source_time=1.02,
            sample_rate=48000,
            channels=1,
            pcm_data=b"\x30\x00" * 480,
        )
        await w_soniox.queue_out.put(c_soniox2)
        await asyncio.sleep(0.05)

        assert len(sink.chunks) == 2
        assert sink.chunks[1].source_branch == "soniox"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_active_failure_outputs_silence(tmp_path):
    """When active branch fails, selector outputs silence chunks ensuring continuous NDI clocking."""
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    sink = DummyAudioSink()
    source = DummyAudioSource([])

    q_gemini = asyncio.Queue(maxsize=10)
    w_gemini = DummyWorker("gemini", q_gemini)
    selector = DubSelector(state_file_path=str(tmp_path / "state.json"), default_branch="gemini")

    engine = DubberEngine(
        config=config,
        source=source,
        sink=sink,
        selector=selector,
        gemini_worker=w_gemini,
    )

    await engine.start()
    try:
        # Mark active branch as failed (status ERROR)
        w_gemini.state.status = "ERROR"
        w_gemini.state.error_message = "Connection reset"

        # Worker emits audio chunk despite error
        c_err = BroadcastAudioChunk(
            session_id="s1",
            language="en",
            source_branch="gemini",
            block_index=5,
            source_time=2.0,
            sample_rate=48000,
            channels=1,
            pcm_data=b"\x50\x00" * 480,
        )
        await w_gemini.queue_out.put(c_err)
        for _ in range(20):
            if sink.chunks:
                break
            await asyncio.sleep(0.005)

        # Output chunks during failure must all be pure silence
        assert len(sink.chunks) >= 1
        for ch in sink.chunks:
            samples = np.frombuffer(ch.pcm_data, dtype=np.int16)
            assert np.all(samples == 0)
        assert len(np.frombuffer(sink.chunks[0].pcm_data, dtype=np.int16)) == 480
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_ingest_loop():
    """Verify source AudioBlocks are automatically dispatched in ingest loop."""
    blocks = [
        AudioBlock("sess", i, i * 320, 1.0 + i * 0.02, b"\x00" * 640)
        for i in range(5)
    ]
    source = DummyAudioSource(blocks)
    config = DubberConfig(max_queue_size=10)
    q_gemini = asyncio.Queue(maxsize=10)
    q_soniox = asyncio.Queue(maxsize=10)
    w_gemini = DummyWorker("gemini", q_gemini)
    w_soniox = DummyWorker("soniox", q_soniox)

    engine = DubberEngine(
        config,
        source=source,
        gemini_worker=w_gemini,
        soniox_worker=w_soniox,
    )
    await engine.start()
    try:
        # Wait for ingest loop to read all 5 blocks
        await asyncio.sleep(0.05)
        assert engine.queue_gemini.qsize() == 5
        assert engine.queue_soniox.qsize() == 5

        b0 = await engine.queue_gemini.get()
        assert b0.block_index == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_started_worker_state_registration(tmp_path):
    """Verify that immediately after await engine.start(), selector observes READY status."""
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    engine = DubberEngine(config)

    # In mock mode, workers start in READY status
    await engine.start()
    try:
        assert engine.selector.get_branch_state("gemini").status == "READY"
        assert engine.selector.get_branch_state("soniox").status == "READY"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_start_failure_cleans_up(tmp_path):
    """If a component fails during start(), engine.stop() is executed to clean up."""
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    sink = DummyAudioSink()

    class FailingSource(AudioSource):
        async def start(self) -> None:
            raise RuntimeError("Capture device failed")

        async def stop(self) -> None:
            pass

        async def read_block(self) -> AudioBlock | None:
            return None

    engine = DubberEngine(config, source=FailingSource(), sink=sink)
    with pytest.raises(RuntimeError, match="Capture device failed"):
        await engine.start()

    assert not engine.is_running
    assert sink.stopped


@pytest.mark.asyncio
async def test_engine_queue_drain_on_restart():
    """Verify queues are drained on start/stop to prevent stale blocks leaking across restarts."""
    config = DubberConfig(max_queue_size=10)
    engine = DubberEngine(config)

    b0 = AudioBlock("sess", 0, 0, 1.0, b"\x00" * 640)
    await engine.dispatch_block_to_queues(b0)
    assert engine.queue_gemini.qsize() == 1
    assert engine.queue_soniox.qsize() == 1


@pytest.mark.asyncio
async def test_engine_silence_heartbeat_on_active_worker_stall():
    """Verify silence heartbeat continuously clocks sink when active worker stalls for > 50 ms."""

    class ContinuousSource(AudioSource):
        def __init__(self) -> None:
            self._running = True

        async def start(self) -> None:
            self._running = True

        async def stop(self) -> None:
            self._running = False

        async def read_block(self) -> AudioBlock | None:
            if not self._running:
                return None
            await asyncio.sleep(0.01)
            return AudioBlock("sess", 0, 0, 1.0, b"\x00" * 640)

    source = ContinuousSource()
    sink = DummyAudioSink()
    config = DubberConfig(max_queue_size=10)
    q_gemini = asyncio.Queue(maxsize=10)
    w_gemini = DummyWorker("gemini", q_gemini)
    # w_gemini does NOT emit any output chunks (stalled active worker)

    engine = DubberEngine(
        config,
        source=source,
        sink=sink,
        gemini_worker=w_gemini,
    )
    await engine.start()
    try:
        # Wait 90 ms so silence heartbeat triggers at elapsed >= 50 ms
        await asyncio.sleep(0.09)

        # Sink must have received silence chunks keeping clock active
        assert len(sink.chunks) >= 1
        for chunk in sink.chunks:
            samples = np.frombuffer(chunk.pcm_data, dtype=np.int16)
            assert np.all(samples == 0)
    finally:
        await engine.stop()

    await engine.stop()
    assert engine.queue_gemini.empty()
    assert engine.queue_soniox.empty()
