"""Core data types and synchronization models for PharmaLive Dubber."""

from dataclasses import dataclass
from typing import Literal

BranchType = Literal["gemini", "soniox"]
BranchStatus = Literal["STARTING", "READY", "RUNNING", "STALE", "ERROR", "STOPPED"]

__all__ = [
    "AudioBlock",
    "BranchState",
    "BranchStatus",
    "BranchType",
    "BroadcastAudioChunk",
]


@dataclass(frozen=True)
class AudioBlock:
    session_id: str
    block_index: int
    sample_offset: int
    source_time: float
    pcm_data: bytes

    @property
    def duration_ms(self) -> float:
        # Assuming 16 kHz 16-bit mono: 2 bytes per sample -> 32 bytes per ms
        return (len(self.pcm_data) / 2) / 16.0


@dataclass(frozen=True)
class BroadcastAudioChunk:
    session_id: str
    language: str
    source_branch: str
    block_index: int
    source_time: float
    sample_rate: int = 48000
    channels: int = 1
    pcm_data: bytes = b""

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0 or self.channels <= 0:
            return 0.0
        bytes_per_sample = 2 * self.channels
        samples = len(self.pcm_data) / bytes_per_sample
        return (samples / self.sample_rate) * 1000.0


@dataclass
class BranchState:
    branch: BranchType
    status: BranchStatus = "STARTING"
    last_audio_time: float = 0.0
    latency_s: float = 0.0
    last_transcript: str = ""
    error_message: str | None = None
    dropped_blocks_count: int = 0

    def is_fresh(self, current_time: float, max_age_s: float = 0.5) -> bool:
        if self.status not in ("READY", "RUNNING"):
            return False
        if self.last_audio_time <= 0.0:
            return False
        age = current_time - self.last_audio_time
        return 0.0 <= age <= max_age_s
