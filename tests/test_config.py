from dubber.config import DubberConfig


def test_default_config():
    config = DubberConfig()
    assert config.source_type == "wav"
    assert config.provider_mode == "mock"
    assert config.ndi_sink_name == "Dub_EN"
    assert config.sample_rate_in == 16000
    assert config.sample_rate_out == 48000
    assert config.max_queue_size == 100
    assert config.freshness_threshold_s == 0.5
    assert config.crossfade_ms == 15
    assert config.soniox_tts_endpoint_url == "wss://tts-rt.soniox.com/tts-websocket"
    assert config.soniox_tts_model == "tts-rt-v2"
    assert config.soniox_tts_voice == "Adrian"

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DUBBER_SOURCE_TYPE", "ndi")
    monkeypatch.setenv("DUBBER_PROVIDER_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("SONIOX_API_KEY", "test_soniox_key")
    monkeypatch.setenv("DUBBER_NDI_SOURCE_NAME", "vMix_Feed")
    monkeypatch.setenv("DUBBER_SONIOX_TTS_ENDPOINT_URL", "wss://custom-tts.soniox.com")
    monkeypatch.setenv("DUBBER_SONIOX_TTS_MODEL", "tts-custom-v1")
    monkeypatch.setenv("DUBBER_SONIOX_TTS_VOICE", "CustomVoice")
    
    config = DubberConfig.from_env()
    assert config.source_type == "ndi"
    assert config.provider_mode == "live"
    assert config.gemini_api_key == "test_gemini_key"
    assert config.soniox_api_key == "test_soniox_key"
    assert config.ndi_source_name == "vMix_Feed"
    assert config.soniox_tts_endpoint_url == "wss://custom-tts.soniox.com"
    assert config.soniox_tts_model == "tts-custom-v1"
    assert config.soniox_tts_voice == "CustomVoice"

def test_config_from_env_empty_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("SONIOX_API_KEY", "")
    monkeypatch.setenv("DUBBER_NDI_SOURCE_NAME", "")
    
    config = DubberConfig.from_env()
    assert config.gemini_api_key is None
    assert config.soniox_api_key is None
    assert config.ndi_source_name is None

