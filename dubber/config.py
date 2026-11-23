import os
from dataclasses import dataclass
from typing import Literal, Optional
from dotenv import load_dotenv

@dataclass
class DubberConfig:
    source_type: Literal["wav", "ndi"] = "wav"
    provider_mode: Literal["mock", "live"] = "mock"
    wav_file_path: Optional[str] = "data/test_speech_ru.wav"
    ndi_source_name: Optional[str] = None
    ndi_sink_name: str = "Dub_EN"
    sample_rate_in: int = 16000
    sample_rate_out: int = 48000
    block_duration_ms: int = 20
    max_queue_size: int = 100
    freshness_threshold_s: float = 0.5
    crossfade_ms: int = 15
    gemini_api_key: Optional[str] = None
    soniox_api_key: Optional[str] = None
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
            state_file_path=os.getenv("DUBBER_STATE_FILE", "state.json"),
            http_port=int(os.getenv("DUBBER_HTTP_PORT", "8000")),
            http_host=os.getenv("DUBBER_HTTP_HOST", "127.0.0.1"),
        )
