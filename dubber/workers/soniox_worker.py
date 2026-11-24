"""Soniox STT v6 + TTS translation worker for PharmaLive Dubber."""

import asyncio
import inspect
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dubber.types import AudioBlock, BranchState, BroadcastAudioChunk
from dubber.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SonioxWorker(BaseWorker):
    """Worker handling speech-to-text translation via Soniox STT v6 and streaming TTS.

    Operates in either 'live' mode (connecting to Soniox WebSocket STT and streaming TTS)
    or 'mock' mode (simulating STT and TTS delays and generating 48 kHz mono audio).
    """

    def __init__(
        self,
        queue_in: asyncio.Queue[AudioBlock],
        mode: Literal["live", "mock"] = "mock",
        transcript_path: str = "logs/transcript_soniox.jsonl",
        simulated_stt_delay_s: float = 0.8,
        simulated_tts_delay_s: float = 0.6,
        target_language: str = "en",
        api_key: str | None = None,
        tts_provider: Callable[[str], Any] | None = None,
        model: str = "stt-rt-v6",
        endpoint_url: str = "wss://api.soniox.com/transcribe-websocket",
        reconnect_delay_s: float = 2.0,
        batch_size: int = 5,
        queue_out_size: int = 100,
        tts_queue_size: int = 10,
        tts_endpoint_url: str = "wss://tts-rt.soniox.com/tts-websocket",
        tts_model: str = "tts-rt-v2",
        tts_voice: str = "Adrian",
    ) -> None:
        self.queue_in: asyncio.Queue[AudioBlock] = queue_in
        self.mode: Literal["live", "mock"] = mode
        self.transcript_path: str = transcript_path
        self.simulated_stt_delay_s: float = simulated_stt_delay_s
        self.simulated_tts_delay_s: float = simulated_tts_delay_s
        self.target_language: str = target_language
        self.api_key: str | None = api_key or os.getenv("SONIOX_API_KEY")
        self.tts_provider: Callable[[str], Any] | None = tts_provider
        self.model: str = model
        self.endpoint_url: str = endpoint_url
        self.reconnect_delay_s: float = reconnect_delay_s
        self.batch_size: int = batch_size
        self.queue_out_size: int = queue_out_size
        self.tts_queue_size: int = tts_queue_size
        self.tts_endpoint_url: str = tts_endpoint_url
        self.tts_model: str = tts_model
        self.tts_voice: str = tts_voice

        parent_dir = Path(self.transcript_path).parent
        if str(parent_dir) not in ("", "."):
            parent_dir.mkdir(parents=True, exist_ok=True)

        self._queue_out: asyncio.Queue[BroadcastAudioChunk | None] = asyncio.Queue(
            maxsize=self.queue_out_size
        )
        self._tts_queue: asyncio.Queue[tuple[str, int, float, float, str]] = (
            asyncio.Queue(maxsize=self.tts_queue_size)
        )
        self._tts_dropped_count: int = 0
        self._state = BranchState(
            branch="soniox",
            status="STARTING",
            last_audio_time=0.0,
            latency_s=0.0,
            last_transcript="",
            error_message=None,
            dropped_blocks_count=0,
        )

        self._stopped: bool = False
        self._worker_task: asyncio.Task[None] | None = None
        self._tts_task: asyncio.Task[None] | None = None
        self._emission_tasks: set[asyncio.Task[None]] = set()

        self._current_session_id: str = "sess"
        self._current_block_index: int = 0
        self._current_source_time: float = 0.0
        self._last_send_time: float = 0.0
        self._last_logged_transcript: str = ""

        # Tracking sent batches for live mode correlation: (session_id, block_index, source_time, send_time)
        self._sent_batches: deque[tuple[str, int, float, float]] = deque(maxlen=500)
        self._mock_sample_count: int = 0

    async def start(self) -> None:
        """Start worker background processing task."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stopped = False
        self._queue_out = asyncio.Queue(maxsize=self.queue_out_size)
        self._tts_queue = asyncio.Queue(maxsize=self.tts_queue_size)
        self._tts_dropped_count = 0
        self._emission_tasks.clear()
        self._sent_batches.clear()
        self._last_logged_transcript = ""
        self._state.status = "READY"
        if self.mode == "mock":
            self._worker_task = asyncio.create_task(self._mock_loop())
        else:
            self._tts_task = asyncio.create_task(
                self._live_tts_consumer(), name="soniox_live_tts_consumer"
            )
            self._worker_task = asyncio.create_task(self._live_loop())

    async def stop(self) -> None:
        """Stop worker background task and release resources."""
        if self._stopped:
            return
        self._stopped = True

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping SonioxWorker task: %s", exc)
            self._worker_task = None

        if self._tts_task is not None:
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping SonioxWorker TTS task: %s", exc)
            self._tts_task = None

        for task in list(self._emission_tasks):
            if not task.done():
                task.cancel()
        if self._emission_tasks:
            try:
                await asyncio.gather(*list(self._emission_tasks), return_exceptions=True)
            except Exception:  # noqa: S110, BLE001
                pass
        self._emission_tasks.clear()

        try:
            self._queue_out.put_nowait(None)
        except asyncio.QueueFull:
            pass

        self._state.status = "STOPPED"

    async def get_output_chunk(self) -> BroadcastAudioChunk | None:
        """Retrieve next translated audio chunk for broadcast."""
        if self._stopped and self._queue_out.empty():
            return None
        try:
            return await self._queue_out.get()
        except asyncio.CancelledError:
            return None

    def get_state(self) -> BranchState:
        """Return the current branch state."""
        return self._state

    def _log_transcript(
        self,
        session_id: str,
        block_index: int,
        source_time: float,
        text: str,
    ) -> None:
        """Log recognized/translated transcript independently to jsonl file."""
        try:
            record = {
                "timestamp": time.time(),
                "branch": "soniox",
                "session_id": session_id,
                "block_index": block_index,
                "source_time": source_time,
                "language": self.target_language,
                "text": text,
            }
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as exc:
            logger.error("Failed to log Soniox transcript: %s", exc)

    async def _mock_loop(self) -> None:
        """Continuous simulation loop for mock mode. Consumes queue without blocking."""
        self._state.status = "READY"
        while not self._stopped:
            try:
                block = await self.queue_in.get()
                batch = [block]
                while len(batch) < self.batch_size:
                    try:
                        next_blk = await asyncio.wait_for(
                            self.queue_in.get(), timeout=0.03
                        )
                        batch.append(next_blk)
                    except TimeoutError:
                        break

                if self._stopped:
                    break

                session_id = batch[0].session_id
                block_index = batch[0].block_index
                source_time = batch[0].source_time

                batch_duration_ms = sum(b.duration_ms for b in batch)
                if batch_duration_ms <= 0:
                    batch_duration_ms = 100.0
                num_samples = int(48000 * (batch_duration_ms / 1000.0))
                if num_samples > 0:
                    t = (self._mock_sample_count + np.arange(num_samples, dtype=np.float32)) / 48000.0
                    self._mock_sample_count += num_samples
                    pcm_data = (np.sin(2.0 * np.pi * 500.0 * t) * (0.2 * 32767)).astype(np.int16).tobytes()
                else:
                    pcm_data = b""

                mock_text = (
                    f"Mock translated speech for session {session_id} block {block_index}"
                )

                chunk = BroadcastAudioChunk(
                    session_id=session_id,
                    language=self.target_language,
                    source_branch="soniox",
                    block_index=block_index,
                    source_time=source_time,
                    sample_rate=48000,
                    channels=1,
                    pcm_data=pcm_data,
                )

                # Schedule delayed emission without blocking the consumer loop
                emit_task = asyncio.create_task(
                    self._delayed_mock_emit(
                        chunk,
                        mock_text,
                        self.simulated_stt_delay_s,
                        self.simulated_tts_delay_s,
                    ),
                    name=f"soniox_mock_emit_{block_index}",
                )
                self._emission_tasks.add(emit_task)
                emit_task.add_done_callback(self._emission_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in SonioxWorker mock loop: %s", exc)
                self._state.status = "ERROR"
                self._state.error_message = str(exc)
                await asyncio.sleep(0.1)

    async def _delayed_mock_emit(
        self,
        chunk: BroadcastAudioChunk,
        mock_text: str,
        stt_delay_s: float,
        tts_delay_s: float,
    ) -> None:
        """Emit text transcript after STT delay and audio chunk after STT + TTS delay."""
        try:
            # 1. STT delay
            if stt_delay_s > 0:
                await asyncio.sleep(stt_delay_s)
            if self._stopped:
                return

            # Emit transcript independently of TTS state
            if mock_text != self._last_logged_transcript:
                self._last_logged_transcript = mock_text
                self._log_transcript(
                    chunk.session_id, chunk.block_index, chunk.source_time, mock_text
                )
                self._state.last_transcript = mock_text

            # 2. TTS delay & synthesis
            if self.tts_provider is not None:
                try:
                    res = self.tts_provider(mock_text)
                    if inspect.isawaitable(res):
                        tts_pcm = await res
                    else:
                        tts_pcm = res
                    if isinstance(tts_pcm, bytes) and len(tts_pcm) > 0:
                        chunk = BroadcastAudioChunk(
                            session_id=chunk.session_id,
                            language=chunk.language,
                            source_branch=chunk.source_branch,
                            block_index=chunk.block_index,
                            source_time=chunk.source_time,
                            sample_rate=48000,
                            channels=1,
                            pcm_data=tts_pcm,
                        )
                    else:
                        return
                except Exception as tts_exc:  # noqa: BLE001
                    logger.error("TTS synthesis failed in SonioxWorker: %s", tts_exc)
                    self._state.status = "ERROR"
                    self._state.error_message = f"TTS error: {tts_exc}"
                    return
            else:
                if tts_delay_s > 0:
                    await asyncio.sleep(tts_delay_s)

            if self._stopped:
                return

            now = time.monotonic()
            total_delay = stt_delay_s + tts_delay_s
            self._state.status = "RUNNING"
            self._state.last_audio_time = now
            self._state.latency_s = total_delay

            if self._queue_out.full():
                try:
                    dropped = self._queue_out.get_nowait()
                    if dropped is not None:
                        logger.warning(
                            "SonioxWorker discarded block %d", dropped.block_index
                        )
                    self._state.dropped_blocks_count += 1
                except asyncio.QueueEmpty:
                    pass
            await self._queue_out.put(chunk)

        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in SonioxWorker delayed emission: %s", exc)

    async def _live_loop(self) -> None:
        """Background loop for live Soniox WebSocket connection."""
        import websockets

        while not self._stopped:
            try:
                self._sent_batches.clear()
                async with websockets.connect(self.endpoint_url) as ws:
                    self._sent_batches.clear()
                    self._state.status = "READY"
                    self._state.error_message = None

                    # Send initial configuration message
                    init_msg: dict[str, Any] = {
                        "model": self.model,
                        "audio_format": "pcm_s16le",
                        "sample_rate": 16000,
                        "num_channels": 1,
                        "translation": {
                            "type": "one_way",
                            "target_language": self.target_language,
                        },
                    }
                    if self.api_key:
                        init_msg["api_key"] = self.api_key

                    await ws.send(json.dumps(init_msg))

                    sender_task = asyncio.create_task(
                        self._live_sender(ws),
                        name="soniox_live_sender",
                    )
                    receiver_task = asyncio.create_task(
                        self._live_receiver(ws),
                        name="soniox_live_receiver",
                    )

                    try:
                        done, _ = await asyncio.wait(
                            [sender_task, receiver_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            exc = task.exception()
                            if exc and not isinstance(exc, asyncio.CancelledError):
                                raise exc
                    finally:
                        for task in (sender_task, receiver_task):
                            if not task.done():
                                task.cancel()
                                try:
                                    await task
                                except (asyncio.CancelledError, Exception):  # noqa: S110, BLE001
                                    pass

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self._sent_batches.clear()
                logger.warning("SonioxWorker live connection error: %s", exc)
                self._state.status = "ERROR"
                self._state.error_message = str(exc)
                await self._drain_queue_during_error(self.reconnect_delay_s)

    async def _live_sender(self, ws: Any) -> None:
        """Send input PCM audio batches to Soniox WebSocket session."""
        while not self._stopped:
            block = await self.queue_in.get()
            batch = [block]
            while len(batch) < self.batch_size:
                try:
                    next_blk = await asyncio.wait_for(
                        self.queue_in.get(), timeout=0.03
                    )
                    batch.append(next_blk)
                except TimeoutError:
                    break

            batch_pcm = b"".join(b.pcm_data for b in batch)
            session_id = batch[-1].session_id
            block_index = batch[-1].block_index
            source_time = batch[-1].source_time
            send_time = time.monotonic()

            self._current_session_id = session_id
            self._current_block_index = block_index
            self._current_source_time = source_time
            self._last_send_time = send_time
            self._sent_batches.append(
                (session_id, block_index, source_time, send_time)
            )

            await ws.send(batch_pcm)

    async def _live_receiver(self, ws: Any) -> None:
        """Receive translated tokens from Soniox and route to TTS synthesizer."""
        async for message in ws:
            if self._stopped:
                break
            try:
                if isinstance(message, bytes):
                    data = json.loads(message.decode("utf-8"))
                else:
                    data = json.loads(message)
            except Exception:  # noqa: BLE001, S112
                continue

            text = ""
            tts_text = ""
            if "tokens" in data and isinstance(data["tokens"], list):
                translated_parts = [
                    t.get("text", "")
                    for t in data["tokens"]
                    if t.get("translation_status") == "translation"
                    or ("translation_status" not in t and t.get("text"))
                ]
                text = "".join(translated_parts).strip()

                final_parts = [
                    t.get("text", "")
                    for t in data["tokens"]
                    if (
                        t.get("translation_status") == "translation"
                        or ("translation_status" not in t and t.get("text"))
                    )
                    and t.get("is_final", True) is True
                ]
                tts_text = "".join(final_parts).strip()
            elif "text" in data:
                text = str(data["text"]).strip()
                tts_text = text if data.get("is_final", True) is True else ""
            elif "translation" in data:
                text = str(data["translation"]).strip()
                tts_text = text if data.get("is_final", True) is True else ""

            if not text:
                continue

            if self._sent_batches:
                session_id, block_index, source_time, send_time = self._sent_batches[0]
            else:
                session_id = self._current_session_id or "session"
                block_index = self._current_block_index
                source_time = self._current_source_time
                send_time = self._last_send_time

            # 1. Log transcript independently of TTS state/failures
            if text != self._last_logged_transcript:
                self._last_logged_transcript = text
                self._state.last_transcript = text
                self._log_transcript(
                    session_id=session_id,
                    block_index=block_index,
                    source_time=source_time,
                    text=text,
                )

            # Only finalized tokens are synthesized to avoid repeating/stuttering audio
            if not tts_text:
                continue

            if self._sent_batches:
                (
                    session_id,
                    block_index,
                    source_time,
                    send_time,
                ) = self._sent_batches.popleft()

            # 2. Queue for decoupled TTS synthesis without blocking STT receiver
            item = (session_id, block_index, source_time, send_time, tts_text)
            if self._tts_queue.full():
                try:
                    self._tts_queue.get_nowait()
                    self._tts_dropped_count += 1
                    self._state.error_message = (
                        f"TTS queue overflow ({self._tts_dropped_count} dropped)"
                    )
                    logger.warning(
                        "SonioxWorker TTS queue full; dropped oldest translation to preserve real-time pace"
                    )
                except asyncio.QueueEmpty:
                    pass

            try:
                self._tts_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    async def _live_tts_consumer(self) -> None:
        """Background task synthesizing queued translation texts into streaming audio."""
        while not self._stopped:
            try:
                item = await self._tts_queue.get()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error reading from TTS queue: %s", exc)
                continue

            session_id, block_index, source_time, send_time, tts_text = item
            try:
                provider = self.tts_provider
                if provider is None:
                    pcm_48k = np.zeros(4800, dtype=np.int16).tobytes()
                    await self._emit_live_pcm_chunk(
                        session_id, block_index, source_time, send_time, pcm_48k
                    )
                    continue

                if hasattr(provider, "synthesize_stream"):
                    async for pcm_chunk in provider.synthesize_stream(tts_text):
                        if self._stopped:
                            break
                        if pcm_chunk:
                            await self._emit_live_pcm_chunk(
                                session_id, block_index, source_time, send_time, pcm_chunk
                            )
                elif inspect.isasyncgenfunction(provider):
                    async for pcm_chunk in provider(tts_text):
                        if self._stopped:
                            break
                        if pcm_chunk:
                            await self._emit_live_pcm_chunk(
                                session_id, block_index, source_time, send_time, pcm_chunk
                            )
                else:
                    res = provider(tts_text)
                    if inspect.isasyncgen(res):
                        async for pcm_chunk in res:
                            if self._stopped:
                                break
                            if pcm_chunk:
                                await self._emit_live_pcm_chunk(
                                    session_id, block_index, source_time, send_time, pcm_chunk
                                )
                    else:
                        if inspect.isawaitable(res):
                            pcm_chunk = await res
                        else:
                            pcm_chunk = res
                        if isinstance(pcm_chunk, bytes) and len(pcm_chunk) > 0:
                            await self._emit_live_pcm_chunk(
                                session_id, block_index, source_time, send_time, pcm_chunk
                            )
            except asyncio.CancelledError:
                break
            except Exception as tts_exc:  # noqa: BLE001
                logger.error("Soniox live TTS synthesis error: %s", tts_exc)
                self._state.error_message = f"TTS error: {tts_exc}"

    async def _emit_live_pcm_chunk(
        self,
        session_id: str,
        block_index: int,
        source_time: float,
        send_time: float,
        pcm_data: bytes,
    ) -> None:
        """Deliver a synthesized PCM chunk to the output queue and update latency/freshness."""
        now = time.monotonic()
        latency_s = max(0.0, now - send_time) if send_time > 0 else 0.0
        self._state.status = "RUNNING"
        self._state.last_audio_time = now
        self._state.latency_s = latency_s
        if self._state.error_message and self._state.error_message.startswith("TTS"):
            self._state.error_message = None

        chunk = BroadcastAudioChunk(
            session_id=session_id,
            language=self.target_language,
            source_branch="soniox",
            block_index=block_index,
            source_time=source_time,
            sample_rate=48000,
            channels=1,
            pcm_data=pcm_data,
        )

        if self._queue_out.full():
            try:
                dropped = self._queue_out.get_nowait()
                if dropped is not None:
                    logger.warning(
                        "SonioxWorker discarded block %d", dropped.block_index
                    )
                self._state.dropped_blocks_count += 1
            except asyncio.QueueEmpty:
                pass
        await self._queue_out.put(chunk)

    async def _drain_queue_during_error(self, duration: float) -> None:
        """Discard incoming blocks during error/reconnection to avoid stale queues."""
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self._stopped:
            while not self.queue_in.empty():
                try:
                    dropped = self.queue_in.get_nowait()
                    self._state.dropped_blocks_count += 1
                    logger.warning(
                        "SonioxWorker discarded block %d", dropped.block_index
                    )
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(min(0.05, max(0.01, deadline - time.monotonic())))
