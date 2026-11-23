"""WAV audio sink for writing 48 kHz mono 16-bit PCM WAV recordings."""

import os
import threading
import wave

from dubber.sink.base import AudioSink
from dubber.types import BroadcastAudioChunk


class WAVAudioSink(AudioSink):
    """Writes BroadcastAudioChunk stream to a 16-bit PCM WAV file."""

    def __init__(self, file_path: str, sample_rate: int = 48000, channels: int = 1) -> None:
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.channels = channels
        self._wf: wave.Wave_write | None = None
        self._lock = threading.Lock()

    async def start(self) -> None:
        dirname = os.path.dirname(os.path.abspath(self.file_path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with self._lock:
            if self._wf is not None:
                self._wf.close()
            self._wf = wave.open(self.file_path, "wb")  # noqa: SIM115
            self._wf.setnchannels(self.channels)
            self._wf.setsampwidth(2)  # 16-bit PCM
            self._wf.setframerate(self.sample_rate)

    async def stop(self) -> None:
        with self._lock:
            if self._wf is not None:
                self._wf.close()
                self._wf = None

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        if not chunk.pcm_data:
            return

        bytes_per_frame = 2 * (self.channels if self.channels > 0 else 1)
        raw_bytes = chunk.pcm_data
        rem = len(raw_bytes) % bytes_per_frame
        if rem != 0:
            raw_bytes = raw_bytes[:-rem]
        if not raw_bytes:
            return

        with self._lock:
            if self._wf is not None:
                self._wf.writeframes(raw_bytes)
