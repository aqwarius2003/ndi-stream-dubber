"""NDI audio source capturing frames and emitting 16 kHz mono 20 ms AudioBlocks."""

import asyncio
import logging
import threading
import time
from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:
    import NDIlib as ndi
except ImportError:
    ndi = None

from dubber.ingest.base import AudioSource
from dubber.types import AudioBlock

logger = logging.getLogger(__name__)


def discover_sources() -> list[str]:
    """Return a snapshot, including senders on this machine."""
    if ndi is None or not ndi.initialize():
        raise RuntimeError("Не удалось загрузить NDI Runtime.")
    finder = ndi.find_create_v2()
    if not finder:
        raise RuntimeError("Не удалось запустить поиск NDI.")
    try:
        deadline = time.monotonic() + 1.6
        while time.monotonic() < deadline:
            ndi.find_wait_for_sources(finder, 150)
        return sorted({s.ndi_name for s in ndi.find_get_current_sources(finder)})
    finally:
        ndi.find_destroy(finder)


class NDIAudioSource(AudioSource):
    """Captures audio frames from an NDI source asynchronously in an executor thread,

    resamples/downmixes to 16 kHz Mono PCM16, buffers and emits 20 ms AudioBlock chunks.
    """

    def __init__(
        self,
        source_name: str | None = None,
        session_id: str = "ndi_session",
        exact_name: bool = False,
    ) -> None:
        self.source_name = source_name
        self.exact_name = exact_name
        self.session_id = session_id
        self._recv = None
        self._running = False
        self._block_index = 0
        self._sample_offset = 0
        self._pcm_buffer = bytearray()
        self._block_bytes = 640  # 320 samples * 2 bytes (20 ms at 16 kHz)
        self._lock = threading.RLock()
        self._capture_mode: str | None = None

    def _reset_state(self) -> None:
        """Reset sequence counters and buffer for a new streaming session."""
        self._block_index = 0
        self._sample_offset = 0
        self._pcm_buffer.clear()

    async def start(self) -> None:
        if ndi is None:
            raise RuntimeError("NDIlib is not installed. Install ndi-python to use NDIAudioSource.")
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDIlib.")

        self._reset_state()
        recv, target_name = await asyncio.to_thread(self._connect)
        with self._lock:
            self._recv = recv
            self._running = True
        logger.info("Connected NDIAudioSource to '%s'", target_name)

    def _connect(self) -> tuple[object, str]:
        """Synchronously discover and connect to target NDI source."""
        find = ndi.find_create_v2()
        if not find:
            raise RuntimeError("Failed to create NDI find instance.")

        try:
            target_source = None
            # Search for source if name given, or take first available
            for _ in range(20):
                ndi.find_wait_for_sources(find, 100)
                sources = ndi.find_get_current_sources(find)
                if sources:
                    if self.source_name:
                        for s in sources:
                            if (s.ndi_name == self.source_name if self.exact_name
                                else self.source_name.lower() in s.ndi_name.lower()):
                                target_source = s
                                break
                    else:
                        target_source = sources[0]
                    if target_source:
                        # The finder owns the source descriptors until it is destroyed.
                        return self._create_receiver(target_source)
        finally:
            ndi.find_destroy(find)

        raise RuntimeError(f"NDI source '{self.source_name or 'any'}' not found on the network.")

    @staticmethod
    def _create_receiver(target_source) -> tuple[object, str]:
        recv_create = ndi.RecvCreateV3()
        recv_create.source_to_connect_to = target_source
        recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        recv_create.bandwidth = ndi.RECV_BANDWIDTH_AUDIO_ONLY
        recv = ndi.recv_create_v3(recv_create)
        if not recv:
            raise RuntimeError(f"Failed to connect to NDI source: {target_source.ndi_name}")

        return recv, target_source.ndi_name

    async def stop(self) -> None:
        self._running = False
        with self._lock:
            if self._recv and ndi:
                ndi.recv_destroy(self._recv)
                self._recv = None

    async def read_block(self) -> AudioBlock | None:
        if not self._running or not self._recv:
            return None

        # Return from buffer if we already have >= 640 bytes
        while len(self._pcm_buffer) < self._block_bytes and self._running:
            # Capture frame in thread executor to not block asyncio event loop
            loop = asyncio.get_running_loop()
            pcm_chunk = await loop.run_in_executor(None, self._capture_pcm)
            if not pcm_chunk:
                await asyncio.sleep(0.005)
                continue

            self._pcm_buffer.extend(pcm_chunk)

        if len(self._pcm_buffer) >= self._block_bytes:
            data = bytes(self._pcm_buffer[: self._block_bytes])
            del self._pcm_buffer[: self._block_bytes]

            block = AudioBlock(
                session_id=self.session_id,
                block_index=self._block_index,
                sample_offset=self._sample_offset,
                source_time=time.monotonic(),
                pcm_data=data,
            )
            self._block_index += 1
            self._sample_offset += 320
            return block

        return None

    def _capture_pcm(self) -> bytes:
        # Keep native frame ownership in the capture thread even if read_block is cancelled.
        with self._lock:
            frame = self._capture_frame()
            if frame is None:
                return b""
            try:
                return self._process_frame_to_pcm16_16k(frame)
            finally:
                self._free_frame(frame)

    def _capture_frame(self):
        with self._lock:
            if not self._running or not self._recv or ndi is None:
                return None

            audio_frame = ndi.AudioFrameV3()
            if self._capture_mode == "pybind11":
                res = ndi.recv_capture_v3(self._recv, 50, False, True, False)
            elif self._capture_mode == "5args":
                res = ndi.recv_capture_v3(self._recv, None, audio_frame, None, 50)
            else:
                try:
                    res = ndi.recv_capture_v3(self._recv, None, audio_frame, None, 50)
                    self._capture_mode = "5args"
                except TypeError:
                    res = ndi.recv_capture_v3(self._recv, 50, False, True, False)
                    self._capture_mode = "pybind11"

            if isinstance(res, tuple):
                frame_type = res[0]
                if frame_type == ndi.FRAME_TYPE_AUDIO:
                    return res[2] if len(res) > 2 and res[2] is not None else audio_frame
                return None
            else:
                if res == ndi.FRAME_TYPE_AUDIO:
                    return audio_frame
                return None

    def _free_frame(self, audio_frame) -> None:
        with self._lock:
            if self._recv and ndi and hasattr(ndi, "recv_free_audio_v3"):
                try:
                    ndi.recv_free_audio_v3(self._recv, audio_frame)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("recv_free_audio_v3 failed: %s", exc)

    def _process_frame_to_pcm16_16k(self, frame) -> bytes:
        data = getattr(frame, "data", None)
        if data is None or getattr(frame, "no_samples", 0) == 0:
            return b""

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.size == 0:
            return b""

        # Normalize input to float32 in [-1.0, 1.0]
        if np.issubdtype(data.dtype, np.floating):
            data_f32 = data.astype(np.float32)
        elif data.dtype == np.int16:
            data_f32 = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data_f32 = data.astype(np.float32) / 2147483648.0
        else:
            data_f32 = data.astype(np.float32) / 32768.0

        # Downmix multi-channel to mono in Float32
        channels = getattr(frame, "no_channels", 1)
        if channels > 1:
            if data_f32.ndim == 2:
                if data_f32.shape[1] == channels:
                    data_f32 = np.mean(data_f32, axis=1)
                elif data_f32.shape[0] == channels:
                    data_f32 = np.mean(data_f32, axis=0)
                else:
                    data_f32 = np.mean(data_f32, axis=1)
            elif data_f32.ndim == 1:
                data_f32 = data_f32.reshape(-1, channels)
                data_f32 = np.mean(data_f32, axis=1)
        elif data_f32.ndim > 1:
            data_f32 = data_f32.squeeze()

        # Resample in Float32 if necessary
        orig_sr = getattr(frame, "sample_rate", 16000)
        if orig_sr != 16000 and len(data_f32) > 0:
            g = gcd(orig_sr, 16000)
            up = 16000 // g
            down = orig_sr // g
            data_f32 = resample_poly(data_f32, up, down)

        # Single conversion to int16 at the end
        data_i16 = np.clip(np.round(data_f32 * 32767.0), -32768, 32767).astype(np.int16)
        return data_i16.tobytes()
