"""NDI audio source capturing frames and emitting 16 kHz mono 20 ms AudioBlocks."""

import asyncio
import logging
from math import gcd
import time
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

try:
    import NDIlib as ndi
except ImportError:
    ndi = None

from dubber.ingest.base import AudioSource
from dubber.types import AudioBlock

logger = logging.getLogger(__name__)


class NDIAudioSource(AudioSource):
    """Captures audio frames from an NDI source asynchronously in an executor thread,

    resamples/downmixes to 16 kHz Mono PCM16, buffers and emits 20 ms AudioBlock chunks.
    """

    def __init__(
        self,
        source_name: Optional[str] = None,
        session_id: str = "ndi_session",
    ) -> None:
        self.source_name = source_name
        self.session_id = session_id
        self._recv = None
        self._running = False
        self._block_index = 0
        self._sample_offset = 0
        self._pcm_buffer = bytearray()
        self._block_bytes = 640  # 320 samples * 2 bytes (20 ms at 16 kHz)

    async def start(self) -> None:
        if ndi is None:
            raise RuntimeError("NDIlib is not installed. Install ndi-python to use NDIAudioSource.")
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDIlib.")

        find = ndi.find_create_v2()
        if not find:
            raise RuntimeError("Failed to create NDI find instance.")

        target_source = None
        # Search for source if name given, or take first available
        for _ in range(20):
            ndi.find_wait_for_sources(find, 100)
            sources = ndi.find_get_current_sources(find)
            if sources:
                if self.source_name:
                    for s in sources:
                        if self.source_name.lower() in s.ndi_name.lower():
                            target_source = s
                            break
                else:
                    target_source = sources[0]
                if target_source:
                    break
        ndi.find_destroy(find)

        if not target_source:
            raise RuntimeError(f"NDI source '{self.source_name or 'any'}' not found on the network.")

        recv_create = ndi.RecvCreateV3()
        recv_create.source_to_connect_to = target_source
        recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        recv_create.bandwidth = ndi.RECV_BANDWIDTH_AUDIO_ONLY
        self._recv = ndi.recv_create_v3(recv_create)
        if not self._recv:
            raise RuntimeError(f"Failed to connect to NDI source: {target_source.ndi_name}")

        self._running = True
        logger.info("Connected NDIAudioSource to '%s'", target_source.ndi_name)

    async def stop(self) -> None:
        self._running = False
        if self._recv and ndi:
            ndi.recv_destroy(self._recv)
            self._recv = None
            if hasattr(ndi, "destroy"):
                ndi.destroy()

    async def read_block(self) -> Optional[AudioBlock]:
        if not self._running or not self._recv:
            return None

        # Return from buffer if we already have >= 640 bytes
        while len(self._pcm_buffer) < self._block_bytes and self._running:
            # Capture frame in thread executor to not block asyncio event loop
            loop = asyncio.get_running_loop()
            audio_frame = await loop.run_in_executor(None, self._capture_frame)
            if audio_frame is None:
                await asyncio.sleep(0.005)
                continue

            # Convert to 16 kHz mono PCM16
            pcm_chunk = self._process_frame_to_pcm16_16k(audio_frame)
            self._pcm_buffer.extend(pcm_chunk)
            self._free_frame(audio_frame)

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

    def _capture_frame(self):
        audio_frame = ndi.AudioFrameV3()
        try:
            res = ndi.recv_capture_v3(self._recv, None, audio_frame, None, 50)
        except TypeError:
            # Native pybind11 signature: (instance, timeout, want_video, want_audio, want_metadata)
            res = ndi.recv_capture_v3(self._recv, 50, False, True, False)

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
        if self._recv and ndi and hasattr(ndi, "recv_free_audio_v3"):
            try:
                ndi.recv_free_audio_v3(self._recv, audio_frame)
            except Exception:
                pass

    def _process_frame_to_pcm16_16k(self, frame) -> bytes:
        data = getattr(frame, "data", None)
        if data is None or getattr(frame, "no_samples", 0) == 0:
            return b""

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.size == 0:
            return b""

        is_float_audio = np.issubdtype(data.dtype, np.floating)

        # Normalize channels to mono
        channels = getattr(frame, "no_channels", 1)
        if channels > 1:
            if data.ndim == 2:
                if data.shape[1] == channels:
                    data = np.mean(data, axis=1)
                elif data.shape[0] == channels:
                    data = np.mean(data, axis=0)
                else:
                    data = np.mean(data, axis=1)
            elif data.ndim == 1:
                data = data.reshape(-1, channels)
                data = np.mean(data, axis=1)
        elif data.ndim > 1:
            data = data.squeeze()

        # Convert float to int16 or clip existing integers
        if is_float_audio:
            data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        elif data.dtype == np.int16:
            data = data.astype(np.int16)
        elif data.dtype == np.int32:
            data = np.clip(data / 65536.0, -32768, 32767).astype(np.int16)
        else:
            data = np.clip(data, -32768, 32767).astype(np.int16)

        # Resample to 16 kHz if necessary
        orig_sr = getattr(frame, "sample_rate", 16000)
        if orig_sr != 16000 and len(data) > 0:
            g = gcd(orig_sr, 16000)
            up = 16000 // g
            down = orig_sr // g
            resampled = resample_poly(data.astype(np.float32), up, down)
            data = np.clip(resampled, -32768, 32767).astype(np.int16)

        return data.tobytes()
