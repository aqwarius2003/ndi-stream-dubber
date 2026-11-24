"""NDI audio sink maintaining a constant NDI publisher at 48 kHz mono."""

import asyncio
import logging
import threading
import time

import numpy as np

try:
    import NDIlib as ndi
except ImportError:
    ndi = None

from dubber.sink.base import AudioSink
from dubber.sink.playout import PlayoutBuffer
from dubber.types import BroadcastAudioChunk

logger = logging.getLogger(__name__)


class NDIAudioSink(AudioSink):
    """Maintains a persistent NDI audio sender (e.g., 'Dub_EN') publishing 48 kHz mono audio."""

    def __init__(
        self,
        sink_name: str = "Dub_EN",
        sample_rate: int = 48000,
        channels: int = 1,
        *,
        realtime: bool = False,
    ) -> None:
        self.sink_name = sink_name
        self.sample_rate = sample_rate
        self.channels = channels
        self._send = None
        self._running = False
        self._lock = threading.Lock()
        self.realtime = realtime
        self.buffer = PlayoutBuffer(sample_rate, channels)
        self._playout: asyncio.Task | None = None
        self.last_voice_time = 0.0
        self.error: str | None = None

    @property
    def dropped_samples(self) -> int:
        return self.buffer.dropped_samples

    async def start(self) -> None:
        if self._playout is not None:
            await self.stop()
        if ndi is None:
            raise RuntimeError("NDIlib is not installed. Install ndi-python to use NDIAudioSink.")
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDIlib.")

        with self._lock:
            if self._send is not None and ndi is not None:
                ndi.send_destroy(self._send)
                self._send = None

            send_settings = ndi.SendCreate()
            send_settings.ndi_name = self.sink_name
            if self.realtime:
                # The application clocks fixed-duration packets. SDK clocking here
                # otherwise sleeps inside send_send_audio_v3 on the capture event loop.
                send_settings.clock_audio = False
            self._send = ndi.send_create(send_settings)
            if not self._send:
                raise RuntimeError(f"Failed to create NDI sender '{self.sink_name}'")
            self._running = True

        if self.realtime:
            self.error = None
            self.last_voice_time = 0.0
            self.buffer = PlayoutBuffer(self.sample_rate, self.channels)
            self._playout = asyncio.create_task(self._play_frames(), name="ndi_playout")

        logger.info("Started constant NDI publisher '%s' (%d Hz)", self.sink_name, self.sample_rate)

    async def stop(self) -> None:
        if self._playout:
            self._playout.cancel()
            try:
                await self._playout
            except asyncio.CancelledError:
                pass
            self._playout = None
        self.buffer.pcm.clear()
        with self._lock:
            self._running = False
            if self._send is not None and ndi is not None:
                ndi.send_destroy(self._send)
                self._send = None

        logger.info("Stopped NDI publisher '%s'", self.sink_name)

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        if not chunk.pcm_data:
            return

        channels = chunk.channels if chunk.channels > 0 else self.channels
        bytes_per_frame = 2 * channels
        raw_bytes = chunk.pcm_data
        rem = len(raw_bytes) % bytes_per_frame
        if rem != 0:
            raw_bytes = raw_bytes[:-rem]
        if not raw_bytes:
            return

        if self.realtime:
            if not self._running:
                return
            if self.error:
                raise RuntimeError(self.error)
            if channels != self.channels or chunk.sample_rate != self.sample_rate:
                raise ValueError("Audio format differs from NDI playout format")
            self.buffer.push(raw_bytes, time.monotonic(), chunk.source_branch)
        else:
            self._send_pcm(raw_bytes, channels)

    async def _play_frames(self) -> None:
        tick = .020
        deadline = time.monotonic()
        try:
            while self._running:
                now = time.monotonic()
                packet = self.buffer.pull(now)
                self._send_pcm(packet, self.channels)
                if any(packet):
                    self.last_voice_time = now
                deadline += tick
                if deadline < time.monotonic() - tick:
                    deadline = time.monotonic() + tick
                await asyncio.sleep(max(0, deadline - time.monotonic()))
        except Exception as exc:
            self.error = str(exc)
            logger.exception("NDI playout failed")

    def _send_pcm(self, raw_bytes: bytes, channels: int) -> None:
        with self._lock:
            if not self._running or self._send is None or ndi is None:
                return

            samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            if samples_int16.size == 0:
                return

            # NDI expects Float32 normalized audio in [-1.0, 1.0]
            samples_float32 = np.clip(samples_int16.astype(np.float32) / 32768.0, -1.0, 1.0)

            if channels > 1:
                samples_2d = np.ascontiguousarray(samples_float32.reshape(-1, channels).T)
            else:
                samples_2d = samples_float32.reshape(1, -1)

            no_samples = samples_2d.shape[1]

            audio_frame = ndi.AudioFrameV3()
            audio_frame.sample_rate = self.sample_rate
            audio_frame.no_channels = channels
            audio_frame.no_samples = no_samples
            audio_frame.data = samples_2d

            ndi.send_send_audio_v3(self._send, audio_frame)
