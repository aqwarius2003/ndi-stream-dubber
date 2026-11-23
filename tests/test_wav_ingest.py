import asyncio
import io
import time
import wave

import numpy as np
import pytest

from dubber.ingest.wav_source import WAVAudioSource
from dubber.types import AudioBlock


def create_synthetic_wav(
    duration_s: float = 0.2,
    sample_rate: int = 48000,
    channels: int = 2,
    sampwidth: int = 2,
) -> bytes:
    buf = io.BytesIO()
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Generate 440 Hz tone
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    if channels == 2:
        audio_data = np.column_stack((audio_data, audio_data)).flatten()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        if sampwidth == 2:
            wf.writeframes(audio_data.tobytes())
        elif sampwidth == 1:
            wf.writeframes(np.zeros(num_samples * channels, dtype=np.uint8).tobytes())
    return buf.getvalue()


@pytest.mark.asyncio
async def test_wav_audio_source_resampling_and_chunking(tmp_path):
    wav_path = tmp_path / "test_48k_mono.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.1, sample_rate=48000, channels=1))

    source = WAVAudioSource(file_path=str(wav_path), session_id="sess_1", real_time_pace=False)
    await source.start()

    blocks = []
    try:
        while True:
            block = await source.read_block()
            if block is None:
                break
            blocks.append(block)
    finally:
        await source.stop()

    # 0.1s at 16 kHz = 1600 samples. At 320 samples per block (20ms), that is 5 blocks.
    assert len(blocks) == 5
    for i, blk in enumerate(blocks):
        assert isinstance(blk, AudioBlock)
        assert blk.block_index == i
        assert blk.sample_offset == i * 320
        assert len(blk.pcm_data) == 640  # 320 samples * 2 bytes
        assert blk.session_id == "sess_1"

    # Reading after stop / EOF returns None
    assert await source.read_block() is None


@pytest.mark.asyncio
async def test_wav_audio_source_stereo_downmix(tmp_path):
    wav_path = tmp_path / "test_stereo.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.1, sample_rate=48000, channels=2))

    source = WAVAudioSource(file_path=str(wav_path), session_id="sess_stereo", real_time_pace=False)
    await source.start()

    blocks = []
    try:
        while True:
            block = await source.read_block()
            if block is None:
                break
            blocks.append(block)
    finally:
        await source.stop()

    assert len(blocks) == 5
    for blk in blocks:
        assert len(blk.pcm_data) == 640


@pytest.mark.asyncio
async def test_wav_audio_source_16k_mono(tmp_path):
    wav_path = tmp_path / "test_16k_mono.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.06, sample_rate=16000, channels=1))

    source = WAVAudioSource(file_path=str(wav_path), session_id="sess_16k", real_time_pace=False)
    await source.start()

    blocks = []
    try:
        while True:
            block = await source.read_block()
            if block is None:
                break
            blocks.append(block)
    finally:
        await source.stop()

    # 0.06s at 16 kHz = 960 samples -> 3 blocks of 320
    assert len(blocks) == 3


@pytest.mark.asyncio
async def test_wav_audio_source_invalid_sample_width(tmp_path):
    wav_path = tmp_path / "test_8bit.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.05, sample_rate=16000, channels=1, sampwidth=1))

    source = WAVAudioSource(file_path=str(wav_path))
    with pytest.raises(ValueError, match="Only 16-bit PCM WAV is supported"):
        await source.start()


@pytest.mark.asyncio
async def test_wav_audio_source_realtime_pacing(tmp_path):
    wav_path = tmp_path / "test_pacing.wav"
    # 3 blocks = 60ms of audio
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.06, sample_rate=16000, channels=1))

    source = WAVAudioSource(file_path=str(wav_path), real_time_pace=True)
    t0 = time.monotonic()
    await source.start()

    count = 0
    try:
        while True:
            block = await source.read_block()
            if block is None:
                break
            count += 1
    finally:
        await source.stop()

    elapsed = time.monotonic() - t0
    assert count == 3
    # Pacing for 3 blocks of 20ms: block 0 (t=0), block 1 (t=20ms), block 2 (t=40ms).
    # Total elapsed should be at least ~0.035s
    assert elapsed >= 0.035


@pytest.mark.asyncio
async def test_wav_audio_source_async_context_manager(tmp_path):
    wav_path = tmp_path / "test_ctx.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.1, sample_rate=16000, channels=1))

    async with WAVAudioSource(file_path=str(wav_path), session_id="ctx_sess", real_time_pace=False) as source:
        block = await source.read_block()
        assert block is not None
        assert block.block_index == 0

    # Outside context manager, source is stopped and read_block returns None
    assert await source.read_block() is None


@pytest.mark.asyncio
async def test_wav_audio_source_stop_while_sleeping(tmp_path):
    wav_path = tmp_path / "test_stop_sleep.wav"
    # 1.0s = 50 blocks
    wav_path.write_bytes(create_synthetic_wav(duration_s=1.0, sample_rate=16000, channels=1))

    source = WAVAudioSource(file_path=str(wav_path), real_time_pace=True)
    await source.start()

    # Read first block
    blk1 = await source.read_block()
    assert blk1 is not None

    # Spawn read_block in background, and stop source concurrently while it's sleeping
    read_task = asyncio.create_task(source.read_block())
    await asyncio.sleep(0.005)
    await source.stop()
    blk2 = await read_task
    assert blk2 is None


@pytest.mark.asyncio
async def test_wav_audio_source_startup_delay_no_burst(tmp_path):
    wav_path = tmp_path / "test_delay.wav"
    wav_path.write_bytes(create_synthetic_wav(duration_s=0.1, sample_rate=16000, channels=1))

    source = WAVAudioSource(file_path=str(wav_path), real_time_pace=True)
    await source.start()

    # Simulate consumer delay before first read
    await asyncio.sleep(0.05)

    t0 = time.monotonic()
    blk0 = await source.read_block()
    assert blk0 is not None
    # Next block should wait ~20ms from first read, not burst immediately
    blk1 = await source.read_block()
    elapsed = time.monotonic() - t0
    assert blk1 is not None
    assert elapsed >= 0.015
    await source.stop()
