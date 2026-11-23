import time

import numpy as np
import pytest

from dubber.selector.selector import DubSelector
from dubber.selector.state import SelectorState
from dubber.types import BranchState, BroadcastAudioChunk


def make_chunk(
    branch: str,
    index: int,
    source_time: float,
    sample_rate: int = 48000,
    channels: int = 1,
    dc_value: int = 1000,
    num_samples: int = 960,
) -> BroadcastAudioChunk:
    pcm = (np.ones(num_samples, dtype=np.int16) * dc_value).tobytes()
    return BroadcastAudioChunk(
        session_id="sess",
        language="en",
        source_branch=branch,
        block_index=index,
        source_time=source_time,
        sample_rate=sample_rate,
        channels=channels,
        pcm_data=pcm,
    )


def test_selector_persistence(tmp_path):
    state_file = str(tmp_path / "state.json")
    sel1 = DubSelector(state_file_path=state_file, default_branch="gemini")
    assert sel1.active_branch == "gemini"

    # Switch to soniox with valid branch state
    sel1.update_branch_state(BranchState("soniox", status="READY", last_audio_time=time.monotonic()))
    ok, msg = sel1.switch_branch("soniox")
    assert ok is True
    assert "soniox" in msg
    assert sel1.active_branch == "soniox"

    # Create new selector, should reload soniox
    sel2 = DubSelector(state_file_path=state_file, default_branch="gemini")
    assert sel2.active_branch == "soniox"


def test_selector_rejects_stale_or_not_ready(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini")

    # Target branch soniox not ready
    selector.update_branch_state(BranchState("soniox", status="ERROR", last_audio_time=time.monotonic()))
    ok, msg = selector.switch_branch("soniox")
    assert ok is False
    assert "NOT READY" in msg

    # Target branch ready, but last audio 0.8s ago (> 0.5s)
    selector.update_branch_state(BranchState("soniox", status="READY", last_audio_time=time.monotonic() - 0.8))
    ok, msg = selector.switch_branch("soniox")
    assert ok is False
    assert "NOT FRESH" in msg


def test_zero_auto_fallback_on_active_failure(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini")

    # Mark gemini as failed
    selector.update_branch_state(BranchState("gemini", status="ERROR", last_audio_time=0.0))
    selector.update_branch_state(BranchState("soniox", status="READY", last_audio_time=time.monotonic()))

    # Active branch remains gemini! No auto-switch!
    assert selector.active_branch == "gemini"
    assert selector.is_alert_active() is True


def test_crossfade_on_switch(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini", crossfade_ms=15)

    # Process gemini chunk (DC offset 1000)
    c1 = make_chunk("gemini", 0, time.monotonic())
    out1 = selector.process_chunk(c1)
    assert out1 is not None

    # Switch to soniox (DC offset 3000)
    selector.update_branch_state(BranchState("soniox", status="READY", last_audio_time=time.monotonic()))
    ok, _ = selector.switch_branch("soniox")
    assert ok is True

    # Next soniox chunk should have crossfaded samples
    c2 = BroadcastAudioChunk(
        "sess",
        "en",
        "soniox",
        1,
        time.monotonic(),
        48000,
        1,
        (np.ones(960, dtype=np.int16) * 3000).tobytes(),
    )
    out2 = selector.process_chunk(c2)
    assert out2 is not None
    samples = np.frombuffer(out2.pcm_data, dtype=np.int16)
    # Start of chunk should be blend of 1000 and 3000 (around 1000 at start, ramping to 3000)
    assert 950 <= samples[0] <= 1100
    assert samples[719] == 3000


def test_inactive_branch_chunk_ignored(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini")
    # Inactive soniox chunk should return None
    c_soniox = make_chunk("soniox", 0, time.monotonic(), dc_value=2000)
    out = selector.process_chunk(c_soniox)
    assert out is None

    # Active gemini chunk should be processed
    c_gemini = make_chunk("gemini", 0, time.monotonic(), dc_value=1000)
    out_active = selector.process_chunk(c_gemini)
    assert out_active is not None
    assert np.frombuffer(out_active.pcm_data, dtype=np.int16)[0] == 1000


def test_active_failure_outputs_silence(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini")
    selector.update_branch_state(BranchState("gemini", status="ERROR", last_audio_time=0.0))

    chunk = make_chunk("gemini", 0, time.monotonic(), dc_value=5000)
    out = selector.process_chunk(chunk)
    assert out is not None
    samples = np.frombuffer(out.pcm_data, dtype=np.int16)
    assert np.all(samples == 0)


def test_corrupted_state_file_fallback(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("invalid json content {{{", encoding="utf-8")

    selector = DubSelector(state_file_path=str(state_file), default_branch="soniox")
    assert selector.active_branch == "soniox"


def test_create_silence_chunk():
    selector = DubSelector(default_branch="gemini", sample_rate=48000, channels=1)
    silence = selector.create_silence_chunk(duration_ms=20.0, session_id="test", language="en")
    assert silence.source_branch == "gemini"
    assert silence.duration_ms == pytest.approx(20.0, abs=0.1)
    samples = np.frombuffer(silence.pcm_data, dtype=np.int16)
    assert len(samples) == 960
    assert np.all(samples == 0)


def test_multi_chunk_crossfade(tmp_path):
    state_file = str(tmp_path / "state.json")
    selector = DubSelector(state_file_path=state_file, default_branch="gemini", crossfade_ms=15)
    # 15 ms at 48000 Hz = 720 samples

    # Process old stream with 360-sample chunks
    c1 = make_chunk("gemini", 0, time.monotonic(), dc_value=1000, num_samples=360)
    out1 = selector.process_chunk(c1)
    assert out1 is not None

    # Switch to soniox
    selector.update_branch_state(BranchState("soniox", status="READY", last_audio_time=time.monotonic()))
    ok, _ = selector.switch_branch("soniox")
    assert ok is True

    # First soniox chunk: 360 samples of 3000 (covers samples 0..359 of fade)
    c2 = make_chunk("soniox", 1, time.monotonic(), dc_value=3000, num_samples=360)
    out2 = selector.process_chunk(c2)
    assert out2 is not None
    s2 = np.frombuffer(out2.pcm_data, dtype=np.int16)
    assert 950 <= s2[0] <= 1050
    assert 1900 <= s2[359] <= 2100

    # Second soniox chunk: 360 samples of 3000 (covers samples 360..719 of fade)
    c3 = make_chunk("soniox", 2, time.monotonic(), dc_value=3000, num_samples=360)
    out3 = selector.process_chunk(c3)
    assert out3 is not None
    s3 = np.frombuffer(out3.pcm_data, dtype=np.int16)
    assert 1950 <= s3[0] <= 2150
    assert s3[359] == 3000


def test_switch_force_and_unknown_branch():
    selector = DubSelector(default_branch="gemini")

    # Unknown branch
    ok, msg = selector.switch_branch("invalid_branch")  # type: ignore[arg-type]
    assert ok is False
    assert "Unknown branch" in msg

    # Switching to already active branch
    ok, msg = selector.switch_branch("gemini")
    assert ok is True
    assert "already active" in msg

    # Switching with force=True bypasses NOT READY
    selector.update_branch_state(BranchState("soniox", status="ERROR", last_audio_time=0.0))
    ok, msg = selector.switch_branch("soniox", force=True)
    assert ok is True
    assert selector.active_branch == "soniox"


def test_selector_state_direct(tmp_path):
    file_path = tmp_path / "direct_state.json"
    state = SelectorState(active_branch="soniox", switch_count=3, metadata={"key": "val"})
    state.save(file_path)

    loaded = SelectorState.load(file_path)
    assert loaded.active_branch == "soniox"
    assert loaded.switch_count == 3
    assert loaded.metadata == {"key": "val"}


@pytest.mark.parametrize("payload", ["[]", "null", '"some_string"', "123", "true"])
def test_selector_state_load_non_dict_json(tmp_path, payload):
    state_file = tmp_path / "nondict.json"
    state_file.write_text(payload, encoding="utf-8")

    state = SelectorState.load(state_file, default_branch="gemini")
    assert state.active_branch == "gemini"
    assert state.switch_count == 0
    assert state.metadata == {}


def test_process_chunk_updates_inactive_branch_freshness(tmp_path):
    selector = DubSelector(default_branch="gemini")

    # Initially soniox is STARTING and last_audio_time is 0.0
    soniox_state = selector.get_branch_state("soniox")
    assert soniox_state.status == "STARTING"
    assert soniox_state.last_audio_time == 0.0

    t_before = time.monotonic()
    c_soniox = make_chunk("soniox", 0, t_before)
    out = selector.process_chunk(c_soniox)
    # Output must be None because soniox is inactive
    assert out is None

    # But its state must now be READY and fresh
    soniox_state_after = selector.get_branch_state("soniox")
    assert soniox_state_after.status == "READY"
    assert soniox_state_after.last_audio_time >= t_before

    # Operator should now be able to switch to soniox cleanly
    ok, _ = selector.switch_branch("soniox")
    assert ok is True
    assert selector.active_branch == "soniox"


def test_process_chunk_odd_bytes_and_empty_chunk(tmp_path):
    selector = DubSelector(default_branch="gemini", crossfade_ms=15)

    # 1. Chunk with DC offset 1000
    c1 = make_chunk("gemini", 0, time.monotonic(), dc_value=1000, num_samples=960)
    out1 = selector.process_chunk(c1)
    assert out1 is not None

    # 2. Empty chunk with 0 samples - should not wipe out _last_tail
    c_empty = BroadcastAudioChunk("sess", "en", "gemini", 1, time.monotonic(), 48000, 1, b"")
    out_empty = selector.process_chunk(c_empty)
    assert out_empty is not None
    assert len(out_empty.pcm_data) == 0

    # 3. Odd-byte length chunk: 960 samples * 2 + 1 byte
    odd_pcm = (np.ones(960, dtype=np.int16) * 1000).tobytes() + b"\xff"
    c_odd = BroadcastAudioChunk("sess", "en", "gemini", 2, time.monotonic(), 48000, 1, odd_pcm)
    out_odd = selector.process_chunk(c_odd)
    assert out_odd is not None
    # Must be trimmed to multiple of 2 bytes
    assert len(out_odd.pcm_data) == 1920
