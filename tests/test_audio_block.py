from dataclasses import FrozenInstanceError

import pytest

from dubber.types import AudioBlock, BranchState, BroadcastAudioChunk


def test_audio_block_creation_and_immutability():
    pcm_bytes = b"\x00\x00" * 320  # 320 samples of 16-bit PCM = 640 bytes (20 ms at 16 kHz)
    block = AudioBlock(
        session_id="test_session",
        block_index=0,
        sample_offset=0,
        source_time=100.0,
        pcm_data=pcm_bytes,
    )
    assert block.block_index == 0
    assert len(block.pcm_data) == 640
    assert block.duration_ms == 20.0

    with pytest.raises(FrozenInstanceError):
        block.block_index = 1  # immutable (frozen)


def test_broadcast_chunk_properties():
    # 48 kHz mono 20 ms chunk = 960 samples = 1920 bytes
    pcm_48k = b"\x00\x00" * 960
    chunk = BroadcastAudioChunk(
        session_id="test_session",
        language="en",
        source_branch="gemini",
        block_index=42,
        source_time=102.5,
        sample_rate=48000,
        channels=1,
        pcm_data=pcm_48k,
    )
    assert chunk.sample_rate == 48000
    assert chunk.duration_ms == 20.0
    assert chunk.source_branch == "gemini"

    with pytest.raises(FrozenInstanceError):
        chunk.block_index = 43  # immutable (frozen)


def test_branch_state_freshness():
    state = BranchState(
        branch="soniox",
        status="READY",
        last_audio_time=100.0,
        latency_s=1.2,
        last_transcript="Hello world",
    )
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is True
    assert state.is_fresh(current_time=100.6, max_age_s=0.5) is False

    # Status must be READY or RUNNING to be fresh
    state.status = "RUNNING"
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is True

    state.status = "STALE"
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is False

    state.status = "ERROR"
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is False

    state.status = "STOPPED"
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is False

    state.status = "STARTING"
    assert state.is_fresh(current_time=100.3, max_age_s=0.5) is False
