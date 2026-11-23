"""Audio output sinks (WAV, NDI)."""

from dubber.sink.base import AudioSink
from dubber.sink.ndi_sink import NDIAudioSink
from dubber.sink.wav_sink import WAVAudioSink

__all__ = ["AudioSink", "NDIAudioSink", "WAVAudioSink"]
