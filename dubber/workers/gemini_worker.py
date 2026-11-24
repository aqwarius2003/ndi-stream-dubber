"""Gemini Live Translate worker for PharmaLive Dubber."""

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.signal import resample_poly

from dubber.types import AudioBlock, BranchState, BroadcastAudioChunk
from dubber.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class GeminiWorker(BaseWorker):
    """Worker handling real-time speech translation via Gemini Live Translate.

    Operates in either 'live' mode (connecting to Gemini Live Translate API)
    or 'mock' mode (simulating translation latency and generating 48 kHz audio).
    """

    def __init__(
        self,
        queue_in: asyncio.Queue[AudioBlock],
        mode: Literal["live", "mock"] = "mock",
        transcript_path: str = "logs/transcript_gemini.jsonl",
        simulated_delay_s: float = 1.2,
        target_language: str = "en",
        api_key: str | None = None,
        model: str = "gemini-3.5-live-translate-preview",
        reconnect_delay_s: float = 2.0,
        batch_size: int = 5,
        queue_out_size: int = 100,
    ) -> None:
        self.queue_in: asyncio.Queue[AudioBlock] = queue_in
        self.mode: Literal["live", "mock"] = mode
        self.transcript_path: str = transcript_path
        self.simulated_delay_s: float = simulated_delay_s
        self.target_language: str = target_language
        self.api_key: str | None = api_key or os.getenv("GEMINI_API_KEY")
        self.model: str = model
        self.reconnect_delay_s: float = reconnect_delay_s
        self.batch_size: int = batch_size
        self.queue_out_size: int = queue_out_size

        parent_dir = Path(self.transcript_path).parent
        if str(parent_dir) not in ("", "."):
            parent_dir.mkdir(parents=True, exist_ok=True)

        self._queue_out: asyncio.Queue[BroadcastAudioChunk | None] = asyncio.Queue(
            maxsize=self.queue_out_size
        )
        self._state = BranchState(
            branch="gemini",
            status="STARTING",
            last_audio_time=0.0,
            latency_s=0.0,
            last_transcript="",
            error_message=None,
            dropped_blocks_count=0,
        )

        self._stopped: bool = False
        self._worker_task: asyncio.Task[None] | None = None
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
        # Reset output queue to clear any leftover sentinel/None from previous stop()
        self._queue_out = asyncio.Queue(maxsize=self.queue_out_size)
        self._emission_tasks.clear()
        self._sent_batches.clear()
        self._last_logged_transcript = ""
        self._state.status = "READY"
        if self.mode == "mock":
            self._worker_task = asyncio.create_task(self._mock_loop())
        else:
            self._worker_task = asyncio.create_task(self._live_loop())

    async def stop(self) -> None:
        """Stop worker background task and release resources."""
        if self._stopped:
            return
        self._stopped = True

        # Stop main worker loop task
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping GeminiWorker task: %s", exc)
            self._worker_task = None

        # Cancel and cleanup any pending delayed emission tasks
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
            chunk = await self._queue_out.get()
            return chunk
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
        try:
            record = {
                "timestamp": time.time(),
                "branch": "gemini",
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
            logger.error("Failed to log transcript: %s", exc)

    async def _mock_loop(self) -> None:
        """Simulation loop for mock mode. Consumes incoming blocks continuously."""
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
                num_samples = int(48000 * (batch_duration_ms / 1000.0))
                if num_samples > 0:
                    t = (self._mock_sample_count + np.arange(num_samples, dtype=np.float32)) / 48000.0
                    self._mock_sample_count += num_samples
                    pcm_data = (np.sin(2.0 * np.pi * 440.0 * t) * (0.2 * 32767)).astype(np.int16).tobytes()
                else:
                    pcm_data = b""

                mock_text = (
                    f"Mock translated speech for session {session_id} block {block_index}"
                )

                chunk = BroadcastAudioChunk(
                    session_id=session_id,
                    language=self.target_language,
                    source_branch="gemini",
                    block_index=block_index,
                    source_time=source_time,
                    sample_rate=48000,
                    channels=1,
                    pcm_data=pcm_data,
                )

                # Schedule delayed emission without blocking the mock consumption loop
                emit_task = asyncio.create_task(
                    self._delayed_emit(chunk, mock_text, self.simulated_delay_s),
                    name=f"mock_emit_{block_index}",
                )
                self._emission_tasks.add(emit_task)
                emit_task.add_done_callback(self._emission_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in GeminiWorker mock loop: %s", exc)
                self._state.status = "ERROR"
                self._state.error_message = str(exc)
                await asyncio.sleep(0.1)

    async def _delayed_emit(
        self,
        chunk: BroadcastAudioChunk,
        mock_text: str,
        delay_s: float,
    ) -> None:
        """Emit translated chunk after simulated processing delay."""
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if self._stopped:
                return

            now = time.monotonic()
            if mock_text != self._last_logged_transcript:
                self._last_logged_transcript = mock_text
                self._log_transcript(
                    chunk.session_id, chunk.block_index, chunk.source_time, mock_text
                )
                self._state.last_transcript = mock_text

            self._state.status = "RUNNING"
            self._state.last_audio_time = now
            self._state.latency_s = delay_s

            if self._queue_out.full():
                try:
                    dropped = self._queue_out.get_nowait()
                    if dropped is not None:
                        logger.warning(
                            "GeminiWorker discarded block %d", dropped.block_index
                        )
                    self._state.dropped_blocks_count += 1
                except asyncio.QueueEmpty:
                    pass
            await self._queue_out.put(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in GeminiWorker delayed emission: %s", exc)

    async def _live_loop(self) -> None:
        """Background loop for live Gemini Live Translate connection."""
        from google import genai
        from google.genai import types

        while not self._stopped:
            try:
                client_kwargs: dict[str, Any] = {
                    "http_options": {"api_version": "v1beta"}
                }
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                client = genai.Client(**client_kwargs)

                config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    input_audio_transcription=types.AudioTranscriptionConfig(),
                    output_audio_transcription=types.AudioTranscriptionConfig(),
                    translation_config=types.TranslationConfig(
                        target_language_code=self.target_language,
                        echo_target_language=False,
                    ),
                )

                async with client.aio.live.connect(
                    model=self.model, config=config
                ) as session:
                    self._state.status = "READY"
                    self._state.error_message = None

                    sender_task = asyncio.create_task(
                        self._live_sender(session, types),
                        name="gemini_live_sender",
                    )
                    receiver_task = asyncio.create_task(
                        self._live_receiver(session),
                        name="gemini_live_receiver",
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
                logger.warning("GeminiWorker live connection error: %s", exc)
                self._state.status = "ERROR"
                self._state.error_message = str(exc)
                await self._drain_queue_during_error(self.reconnect_delay_s)

    async def _live_sender(self, session: Any, types_module: Any) -> None:
        """Send input PCM audio batches to Gemini Live session."""
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

            blob = types_module.Blob(
                data=batch_pcm, mime_type="audio/pcm;rate=16000"
            )
            if hasattr(session, "send_realtime_input"):
                await session.send_realtime_input(audio=blob)
            else:
                await session.send(
                    input={
                        "realtime_input": {
                            "media_chunks": [
                                {
                                    "data": batch_pcm,
                                    "mime_type": "audio/pcm;rate=16000",
                                }
                            ]
                        }
                    }
                )

    async def _live_receiver(self, session: Any) -> None:
        """Receive translated audio and transcription from Gemini Live session."""
        async for response in session.receive():
            if self._stopped:
                break
            server_content = getattr(response, "server_content", None)
            if not server_content:
                continue

            # Check for transcription with deduplication
            text: str | None = None
            output_tx = getattr(server_content, "output_transcription", None)
            if output_tx and getattr(output_tx, "text", None):
                text = output_tx.text.strip()
            elif getattr(server_content, "model_turn", None) and getattr(
                server_content.model_turn, "parts", None
            ):
                for part in server_content.model_turn.parts:
                    if getattr(part, "text", None) and part.text.strip():
                        text = part.text.strip()
                        break

            if text and text != self._last_logged_transcript:
                self._last_logged_transcript = text
                self._state.last_transcript = text
                self._log_transcript(
                    session_id=self._current_session_id or "session",
                    block_index=self._current_block_index,
                    source_time=self._current_source_time,
                    text=text,
                )

            # Check for 24 kHz audio parts
            model_turn = getattr(server_content, "model_turn", None)
            if model_turn and getattr(model_turn, "parts", None):
                for part in model_turn.parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        raw_24k = inline_data.data
                        if len(raw_24k) % 2 != 0:
                            raw_24k = raw_24k[: len(raw_24k) - (len(raw_24k) % 2)]
                        audio_24k = np.frombuffer(raw_24k, dtype=np.int16)
                        if len(audio_24k) == 0:
                            continue

                        # Resample 24k -> 48k: resample_poly(data, 2, 1)
                        resampled_48k = resample_poly(audio_24k, 2, 1)
                        resampled_pcm = (
                            np.clip(resampled_48k, -32768, 32767)
                            .astype(np.int16)
                            .tobytes()
                        )

                        now = time.monotonic()
                        if self._sent_batches:
                            (
                                session_id,
                                block_index,
                                source_time,
                                send_time,
                            ) = self._sent_batches.popleft()
                            latency_s = max(0.0, now - send_time)
                        else:
                            session_id = self._current_session_id or "session"
                            block_index = self._current_block_index
                            source_time = self._current_source_time
                            latency_s = (
                                max(0.0, now - self._last_send_time)
                                if self._last_send_time > 0
                                else 0.0
                            )

                        self._state.status = "RUNNING"
                        self._state.last_audio_time = now
                        self._state.latency_s = latency_s

                        chunk = BroadcastAudioChunk(
                            session_id=session_id,
                            language=self.target_language,
                            source_branch="gemini",
                            block_index=block_index,
                            source_time=source_time,
                            sample_rate=48000,
                            channels=1,
                            pcm_data=resampled_pcm,
                        )

                        if self._queue_out.full():
                            try:
                                dropped = self._queue_out.get_nowait()
                                if dropped is not None:
                                    logger.warning(
                                        "GeminiWorker discarded block %d",
                                        dropped.block_index,
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
                        "GeminiWorker discarded block %d", dropped.block_index
                    )
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(min(0.05, max(0.01, deadline - time.monotonic())))
