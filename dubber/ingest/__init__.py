"""Audio ingest sources (WAV, NDI)."""

from dubber.ingest.base import AudioSource
from dubber.ingest.wav_source import WAVAudioSource

__all__ = ["AudioSource", "WAVAudioSource"]
