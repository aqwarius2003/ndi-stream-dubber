"""NDI audio sink maintaining a constant NDI publisher at 48 kHz mono."""

import logging
import threading

import numpy as np

try:
    import NDIlib as ndi
except ImportError:
    ndi = None

from dubber.sink.base import AudioSink
from dubber.types import BroadcastAudioChunk

logger = logging.getLogger(__name__)


class NDIAudioSink(AudioSink):
    """Maintains a persistent NDI audio sender (e.g., 'Dub_EN') publishing 48 kHz mono audio."""

    def __init__(
        self,
        sink_name: str = "Dub_EN",
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        self.sink_name = sink_name
        self.sample_rate = sample_rate
        self.channels = channels
        self._send = None
        self._running = False
        self._lock = threading.Lock()

    async def start(self) -> None:
        if ndi is None:
            raise RuntimeError("NDIlib is not installed. Install ndi-python to use NDIAudioSink.")
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDIlib.")

        with self._lock:
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = self.sink_name
            self._send = ndi.send_create(send_settings)
            if not self._send:
                raise RuntimeError(f"Failed to create NDI sender '{self.sink_name}'")
            self._running = True

        logger.info("Started constant NDI publisher '%s' (%d Hz)", self.sink_name, self.sample_rate)

    async def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._send is not None and ndi is not None:
                ndi.send_destroy(self._send)
                self._send = None

        logger.info("Stopped NDI publisher '%s'", self.sink_name)

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        if not chunk.pcm_data:
            return

        with self._lock:
            if not self._running or self._send is None or ndi is None:
                return

            samples_int16 = np.frombuffer(chunk.pcm_data, dtype=np.int16)
            if samples_int16.size == 0:
                return

            # NDI expects Float32 normalized audio in [-1.0, 1.0]
            samples_float32 = samples_int16.astype(np.float32) / 32767.0

            channels = chunk.channels if chunk.channels > 0 else self.channels
            if channels > 1:
                samples_2d = samples_float32.reshape(-1, channels).T
            else:
                samples_2d = samples_float32.reshape(1, -1)

            no_samples = samples_2d.shape[1]

            audio_frame = ndi.AudioFrameV3()
            audio_frame.sample_rate = self.sample_rate
            audio_frame.no_channels = channels
            audio_frame.no_samples = no_samples
            audio_frame.data = samples_2d

            ndi.send_send_audio_v3(self._send, audio_frame)
