import os
import pytest
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

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DUBBER_SOURCE_TYPE", "ndi")
    monkeypatch.setenv("DUBBER_PROVIDER_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("SONIOX_API_KEY", "test_soniox_key")
    monkeypatch.setenv("DUBBER_NDI_SOURCE_NAME", "vMix_Feed")
    
    config = DubberConfig.from_env()
    assert config.source_type == "ndi"
    assert config.provider_mode == "live"
    assert config.gemini_api_key == "test_gemini_key"
    assert config.soniox_api_key == "test_soniox_key"
    assert config.ndi_source_name == "vMix_Feed"

def test_config_from_env_empty_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("SONIOX_API_KEY", "")
    monkeypatch.setenv("DUBBER_NDI_SOURCE_NAME", "")
    
    config = DubberConfig.from_env()
    assert config.gemini_api_key is None
    assert config.soniox_api_key is None
    assert config.ndi_source_name is None

