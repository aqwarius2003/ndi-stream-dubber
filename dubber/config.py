import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


@dataclass
class DubberConfig:
    source_type: Literal["wav", "ndi"] = "wav"
    provider_mode: Literal["mock", "live"] = "mock"
    wav_file_path: str | None = "data/test_speech_ru.wav"
    ndi_source_name: str | None = None
    ndi_sink_name: str = "Dub_EN"
    sample_rate_in: int = 16000
    sample_rate_out: int = 48000
    block_duration_ms: int = 20
    max_queue_size: int = 100
    freshness_threshold_s: float = 0.5
    crossfade_ms: int = 15
    gemini_api_key: str | None = None
    soniox_api_key: str | None = None
    soniox_tts_endpoint_url: str = "wss://tts-rt.soniox.com/tts-websocket"
    soniox_tts_model: str = "tts-rt-v2"
    soniox_tts_voice: str = "Adrian"
    state_file_path: str = "state.json"
    http_port: int = 8000
    http_host: str = "127.0.0.1"

    @classmethod
    def from_env(cls) -> "DubberConfig":
        load_dotenv()
        return cls(
            source_type=os.getenv("DUBBER_SOURCE_TYPE", "wav"),  # type: ignore
            provider_mode=os.getenv("DUBBER_PROVIDER_MODE", "mock"),  # type: ignore
            wav_file_path=os.getenv("DUBBER_WAV_FILE", "data/test_speech_ru.wav"),
            ndi_source_name=os.getenv("DUBBER_NDI_SOURCE_NAME") or None,
            ndi_sink_name=os.getenv("DUBBER_NDI_SINK_NAME", "Dub_EN"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            soniox_api_key=os.getenv("SONIOX_API_KEY") or None,
            soniox_tts_endpoint_url=os.getenv(
                "DUBBER_SONIOX_TTS_ENDPOINT_URL", "wss://tts-rt.soniox.com/tts-websocket"
            ),
            soniox_tts_model=os.getenv("DUBBER_SONIOX_TTS_MODEL", "tts-rt-v2"),
            soniox_tts_voice=os.getenv("DUBBER_SONIOX_TTS_VOICE", "Adrian"),
            state_file_path=os.getenv("DUBBER_STATE_FILE", "state.json"),
            http_port=int(os.getenv("DUBBER_HTTP_PORT", "8000")),
            http_host=os.getenv("DUBBER_HTTP_HOST", "127.0.0.1"),
        )
