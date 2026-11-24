"""Packet arrival timing and fixed-duration PCM playout for a live NDI sender."""

from collections import deque
from math import ceil


class PlayoutBuffer:
    tick_ms = 20
    min_ms = 60
    max_ms = 320

    def __init__(self, rate: int, channels: int) -> None:
        self.rate = rate
        self.stride = channels * 2
        self.frame_bytes = rate * self.tick_ms // 1000 * self.stride
        self.silence = bytes(self.frame_bytes)
        self.pcm = bytearray()
        self.target_ms = self.min_ms
        self.dropped_samples = 0
        self.local_silence_samples = 0
        self.provider_zero_samples = 0
        self._late_ms: deque[int] = deque(maxlen=40)
        self._branch: str | None = None
        self._next_arrival: float | None = None
        self._resume_at: float | None = None

    def push(self, data: bytes, now: float, branch: str) -> None:
        if branch != self._branch:
            self.pcm.clear()
            self._late_ms.clear()
            self.target_ms = self.min_ms
            self._next_arrival = self._resume_at = None
            self._branch = branch
        if self._next_arrival is not None:
            delay = round((now - self._next_arrival) * 1000)
            # A gap between utterances must not train the network jitter estimate.
            if delay <= 1000:
                self._late_ms.append(max(delay, 0))
                ordered = sorted(self._late_ms)
                p95 = ordered[ceil(.95 * len(ordered)) - 1]
                wanted = min(self.max_ms, max(self.min_ms, self.tick_ms * (ceil(p95 / self.tick_ms) + 1)))
                self.target_ms = max(wanted, self.target_ms - self.tick_ms)
        self._next_arrival = now + len(data) / (self.rate * self.stride)
        if self._resume_at is None:
            self._resume_at = now + self.target_ms / 1000
        self.pcm.extend(data)
        excess = len(self.pcm) - 2 * self.rate * self.stride
        if excess > 0:
            del self.pcm[:excess]
            self.dropped_samples += excess // self.stride

    def pull(self, now: float) -> bytes:
        if self._branch is None:
            return self.silence
        if self._resume_at is not None and now < self._resume_at:
            self.local_silence_samples += self.frame_bytes // self.stride
            return self.silence
        frame = bytes(self.pcm[:self.frame_bytes])
        del self.pcm[:len(frame)]
        if frame and not any(frame):
            self.provider_zero_samples += len(frame) // self.stride
        if len(frame) < self.frame_bytes:
            self._resume_at = None
            self.local_silence_samples += (self.frame_bytes - len(frame)) // self.stride
            return frame.ljust(self.frame_bytes, b"\x00")
        return frame
