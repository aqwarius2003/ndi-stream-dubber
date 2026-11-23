"""WAV audio source generating real-time paced 16 kHz mono 20 ms AudioBlock chunks."""

import asyncio
import time
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from dubber.ingest.base import AudioSource
from dubber.types import AudioBlock


class WAVAudioSource(AudioSource):
    """Reads a WAV file, converts to 16 kHz mono 16-bit PCM, and yields 20 ms AudioBlocks."""

    def __init__(
        self,
        file_path: str,
        session_id: str = "wav_session",
        real_time_pace: bool = True,
    ) -> None:
        self.file_path = file_path
        self.session_id = session_id
        self.real_time_pace = real_time_pace
        self._block_index = 0
        self._sample_offset = 0
        self._audio_data_16k: np.ndarray = np.array([], dtype=np.int16)
        self._current_pos = 0
        self._start_time: float = 0.0
        self._running = False
        self._block_samples = 320  # 20 ms at 16 kHz (320 samples * 2 bytes = 640 bytes)

    async def start(self) -> None:
        with wave.open(self.file_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            orig_sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}")

        raw_samples = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            raw_samples = raw_samples.reshape(-1, channels)
            raw_samples = np.mean(raw_samples, axis=1).astype(np.int16)

        if orig_sr != 16000:
            # Resample to 16000 Hz using rational fraction resample_poly
            g = gcd(orig_sr, 16000)
            up = 16000 // g
            down = orig_sr // g
            resampled = resample_poly(raw_samples.astype(np.float32), up, down)
            self._audio_data_16k = np.clip(resampled, -32768, 32767).astype(np.int16)
        else:
            self._audio_data_16k = raw_samples

        self._current_pos = 0
        self._block_index = 0
        self._sample_offset = 0
        self._running = True
        self._start_time = time.monotonic()

    async def stop(self) -> None:
        self._running = False

    async def read_block(self) -> AudioBlock | None:
        if not self._running:
            return None

        if self._current_pos + self._block_samples > len(self._audio_data_16k):
            # EOF reached
            self._running = False
            return None

        # Real-time pacing with drift compensation against monotonic clock
        if self.real_time_pace:
            expected_time = self._start_time + (self._sample_offset / 16000.0)
            now = time.monotonic()
            wait_time = expected_time - now
            if wait_time > 0.001:
                await asyncio.sleep(wait_time)

        chunk = self._audio_data_16k[self._current_pos : self._current_pos + self._block_samples]
        block = AudioBlock(
            session_id=self.session_id,
            block_index=self._block_index,
            sample_offset=self._sample_offset,
            source_time=time.monotonic(),
            pcm_data=chunk.tobytes(),
        )

        self._current_pos += self._block_samples
        self._sample_offset += self._block_samples
        self._block_index += 1
        return block
