"""DubberEngine: queue dispatcher and concurrent pipeline orchestration.

Coordinates the complete audio translation pipeline:
- Non-blocking ingest loop reading AudioBlock from AudioSource.
- Dual-queue distribution with drop-oldest policy preserving real-time pace.
- Sibling worker isolation (slow worker never blocks capture or sibling).
- Concurrent worker output processing and routing via DubSelector to AudioSink.
- Continuous audio clocking with silence on active branch failure.
"""

import asyncio
import logging
import time
from types import TracebackType
from typing import Any, Self

from dubber.config import DubberConfig
from dubber.ingest.base import AudioSource
from dubber.selector.selector import DubSelector
from dubber.sink.base import AudioSink
from dubber.types import AudioBlock, BranchType, BroadcastAudioChunk
from dubber.workers.base import BaseWorker
from dubber.workers.gemini_worker import GeminiWorker
from dubber.workers.soniox_tts import SonioxStreamingTTS
from dubber.workers.soniox_worker import SonioxWorker

logger = logging.getLogger(__name__)


class DubberEngine:
    """Central engine orchestrating ingest, worker queues, selector, and audio sink."""

    def __init__(
        self,
        config: DubberConfig | None = None,
        source: AudioSource | None = None,
        sink: AudioSink | None = None,
        selector: DubSelector | None = None,
        gemini_worker: GeminiWorker | BaseWorker | None = None,
        soniox_worker: SonioxWorker | BaseWorker | None = None,
    ) -> None:
        self.config = config or DubberConfig()
        self.source = source
        self.sink = sink

        # Initialize queues and workers with sibling isolation
        if gemini_worker is not None:
            self.gemini_worker: GeminiWorker | BaseWorker = gemini_worker
            self.queue_gemini: asyncio.Queue[AudioBlock] = getattr(
                gemini_worker, "queue_in", None
            ) or asyncio.Queue(maxsize=self.config.max_queue_size)
        else:
            self.queue_gemini = asyncio.Queue(maxsize=self.config.max_queue_size)
            self.gemini_worker = GeminiWorker(
                queue_in=self.queue_gemini,
                mode=self.config.provider_mode,
                api_key=self.config.gemini_api_key,
            )

        if soniox_worker is not None:
            self.soniox_worker: SonioxWorker | BaseWorker = soniox_worker
            self.queue_soniox: asyncio.Queue[AudioBlock] = getattr(
                soniox_worker, "queue_in", None
            ) or asyncio.Queue(maxsize=self.config.max_queue_size)
        else:
            self.queue_soniox = asyncio.Queue(maxsize=self.config.max_queue_size)
            tts_provider = None
            if self.config.provider_mode == "live":
                tts_provider = SonioxStreamingTTS(
                    api_key=self.config.soniox_api_key,
                    endpoint_url=self.config.soniox_tts_endpoint_url,
                    model=self.config.soniox_tts_model,
                    voice=self.config.soniox_tts_voice,
                )
            self.soniox_worker = SonioxWorker(
                queue_in=self.queue_soniox,
                mode=self.config.provider_mode,
                api_key=self.config.soniox_api_key,
                tts_provider=tts_provider,
                tts_endpoint_url=self.config.soniox_tts_endpoint_url,
                tts_model=self.config.soniox_tts_model,
                tts_voice=self.config.soniox_tts_voice,
            )

        # DubSelector managing zero auto-fallback routing and freshness
        if selector is not None:
            self.selector = selector
        else:
            self.selector = DubSelector(
                state_file_path=self.config.state_file_path,
                freshness_threshold_s=self.config.freshness_threshold_s,
                crossfade_ms=self.config.crossfade_ms,
                sample_rate=self.config.sample_rate_out,
            )

        self.dropped_blocks: dict[str, int] = {"gemini": 0, "soniox": 0}
        self._running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_sink_write_time: float = 0.0
        self._source_active: bool = False
        self._sink_lock: asyncio.Lock = asyncio.Lock()

        # Immediately register worker states with the selector
        self._sync_worker_states_to_selector()

    @property
    def is_running(self) -> bool:
        """Return True if background pipeline tasks are currently running."""
        return self._running

    @property
    def is_source_active(self) -> bool:
        """Return True if the audio source is present and capturing."""
        if self.source is None:
            return False
        if hasattr(self.source, "is_active") and callable(self.source.is_active):
            return bool(self.source.is_active())
        if hasattr(self.source, "_running"):
            return bool(self.source._running)
        return self._source_active

    def _sync_worker_states_to_selector(self) -> None:
        """Register worker state objects with the selector."""
        if self.gemini_worker and hasattr(self.gemini_worker, "get_state"):
            self.selector.update_branch_state(self.gemini_worker.get_state())
        if self.soniox_worker and hasattr(self.soniox_worker, "get_state"):
            self.selector.update_branch_state(self.soniox_worker.get_state())

    def _record_drop(self, name: BranchType, dropped_block_index: int) -> None:
        """Record a dropped block on engine, selector, and worker drop metrics."""
        self.dropped_blocks[name] = self.dropped_blocks.get(name, 0) + 1
        logger.warning(
            "Queue overflow on branch '%s'. Dropped block_index=%d to preserve real-time pace.",
            name,
            dropped_block_index,
        )
        if self.selector is not None:
            branch_state = self.selector.get_branch_state(name)
            branch_state.dropped_blocks_count = self.dropped_blocks[name]

        worker = self.gemini_worker if name == "gemini" else self.soniox_worker
        if worker is not None and hasattr(worker, "get_state"):
            worker.get_state().dropped_blocks_count = self.dropped_blocks[name]

    def _drain_queues(self) -> None:
        """Drain worker queues to prevent stale blocks from leaking across restarts."""
        for q in (self.queue_gemini, self.queue_soniox):
            if q is not None:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break

    def _dispatch_to_branch(
        self, name: BranchType, queue: asyncio.Queue[AudioBlock], block: AudioBlock
    ) -> None:
        """Dispatch block to a branch queue applying drop-oldest policy on overflow."""
        if queue.full():
            try:
                dropped = queue.get_nowait()
                self._record_drop(name, dropped.block_index)
            except asyncio.QueueEmpty:
                pass

        try:
            queue.put_nowait(block)
        except asyncio.QueueFull:
            # Fallback if queue became full between check and put
            try:
                dropped = queue.get_nowait()
                self._record_drop(name, dropped.block_index)
                queue.put_nowait(block)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def dispatch_block_to_queues(self, block: AudioBlock) -> None:
        """Dispatches an AudioBlock to both branches with drop-oldest policy.

        Non-blocking: slow or stalled worker NEVER delays ingest or sibling worker.
        """
        for name, q in (("gemini", self.queue_gemini), ("soniox", self.queue_soniox)):
            if q is not None:
                self._dispatch_to_branch(name, q, block)

    async def _ingest_loop(self) -> None:
        """Read AudioBlocks continuously from source and distribute to branch queues."""
        try:
            self._source_active = True
            while self._running and self.source is not None:
                block = await self.source.read_block()
                if block is None:
                    self._source_active = False
                    logger.info("Ingest source finished or stopped.")
                    break
                self._source_active = True
                await self.dispatch_block_to_queues(block)
        except asyncio.CancelledError:
            pass
        except Exception:
            self._source_active = False
            logger.exception("Error in DubberEngine ingest loop")
        finally:
            self._source_active = False

    async def _worker_output_loop(self, worker: GeminiWorker | SonioxWorker | BaseWorker) -> None:
        """Consume translated audio chunks from worker, update selector state, and route to sink."""
        try:
            while self._running:
                chunk: BroadcastAudioChunk | None = await worker.get_output_chunk()
                if chunk is None:
                    break

                # Synchronize latest worker metrics to DubSelector
                if hasattr(worker, "get_state"):
                    self.selector.update_branch_state(worker.get_state())

                # Route chunk through DubSelector (applies active branch filter, crossfade, silence on failure)
                selected_chunk = self.selector.process_chunk(chunk)
                if selected_chunk is not None and self.sink is not None:
                    async with self._sink_lock:
                        await self.sink.write_chunk(selected_chunk)
                        self._last_sink_write_time = time.monotonic()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in worker output loop")

    async def _silence_heartbeat_loop(self) -> None:
        """Continuously generate silence chunks to keep sink clocked during stalls/failures."""
        try:
            while self._running:
                await asyncio.sleep(0.02)
                if not self._running:
                    break

                self._sync_worker_states_to_selector()

                now = time.monotonic()
                elapsed = now - self._last_sink_write_time
                source_active = self.is_source_active
                alert_active = self.selector.is_alert_active()

                if (
                    elapsed >= 0.05
                    and (source_active or alert_active)
                    and self.sink is not None
                ):
                    silence_chunk = self.selector.create_silence_chunk(
                        duration_ms=20.0, session_id="silence"
                    )
                    async with self._sink_lock:
                        await self.sink.write_chunk(silence_chunk)
                        self._last_sink_write_time = time.monotonic()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in DubberEngine silence heartbeat loop")

    async def start(self) -> None:
        """Start workers, sources, sinks, and background orchestration loops."""
        if self._running:
            return
        self._running = True
        try:
            self._drain_queues()
            self._last_sink_write_time = time.monotonic()
            self._source_active = self.source is not None

            if self.source is not None:
                await self.source.start()
            if self.sink is not None:
                await self.sink.start()

            if self.gemini_worker is not None:
                await self.gemini_worker.start()
            if self.soniox_worker is not None:
                await self.soniox_worker.start()

            # Update selector immediately with live states of started workers
            self._sync_worker_states_to_selector()

            self._tasks.append(
                asyncio.create_task(self._ingest_loop(), name="dubber_ingest_loop")
            )
            self._tasks.append(
                asyncio.create_task(
                    self._silence_heartbeat_loop(),
                    name="dubber_silence_heartbeat_loop",
                )
            )
            if self.gemini_worker is not None:
                self._tasks.append(
                    asyncio.create_task(
                        self._worker_output_loop(self.gemini_worker),
                        name="dubber_gemini_output_loop",
                    )
                )
            if self.soniox_worker is not None:
                self._tasks.append(
                    asyncio.create_task(
                        self._worker_output_loop(self.soniox_worker),
                        name="dubber_soniox_output_loop",
                    )
                )

            logger.info("DubberEngine started successfully.")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Safely stop all background orchestration loops, workers, and I/O streams."""
        if not self._running and not self._tasks:
            self._drain_queues()
            return
        self._running = False

        # Cancel and wait for background loops
        for t in self._tasks:
            if not t.done():
                t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        # Stop workers
        if self.gemini_worker is not None:
            await self.gemini_worker.stop()
        if self.soniox_worker is not None:
            await self.soniox_worker.stop()

        # Stop source and sink
        if self.source is not None:
            await self.source.stop()
        if self.sink is not None:
            await self.sink.stop()

        self._drain_queues()
        logger.info("DubberEngine stopped.")

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()
