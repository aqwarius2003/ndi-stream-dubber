"""Unit and integration tests for AudioSink interface, WAVAudioSink, and NDIAudioSink."""

import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dubber.sink.base import AudioSink
from dubber.sink.ndi_sink import NDIAudioSink
from dubber.sink.wav_sink import WAVAudioSink
from dubber.types import BroadcastAudioChunk


@pytest.mark.asyncio
async def test_wav_audio_sink(tmp_path):
    out_wav = tmp_path / "output.wav"
    sink = WAVAudioSink(file_path=str(out_wav), sample_rate=48000, channels=1)
    await sink.start()

    # Send two 20 ms 48 kHz chunks (960 samples * 2 = 1920 bytes each)
    pcm_data = (np.ones(960, dtype=np.int16) * 1000).tobytes()
    chunk1 = BroadcastAudioChunk("sess_1", "en", "gemini", 0, 100.0, 48000, 1, pcm_data)
    chunk2 = BroadcastAudioChunk("sess_1", "en", "gemini", 1, 100.02, 48000, 1, pcm_data)

    await sink.write_chunk(chunk1)
    await sink.write_chunk(chunk2)
    await sink.stop()

    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 1920


@pytest.mark.asyncio
async def test_wav_audio_sink_context_manager(tmp_path):
    out_wav = tmp_path / "nested" / "output_cm.wav"
    pcm_data = (np.ones(480, dtype=np.int16) * 500).tobytes()
    chunk = BroadcastAudioChunk("sess_1", "en", "soniox", 0, 1.0, 48000, 1, pcm_data)

    async with WAVAudioSink(file_path=str(out_wav), sample_rate=48000, channels=1) as sink:
        assert isinstance(sink, AudioSink)
        await sink.write_chunk(chunk)

    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 480


@pytest.mark.asyncio
async def test_wav_audio_sink_noop_when_not_started(tmp_path):
    out_wav = tmp_path / "unopened.wav"
    sink = WAVAudioSink(file_path=str(out_wav))
    # Writing chunk before start() should safely no-op without crashing
    chunk = BroadcastAudioChunk("sess", "en", "soniox", 0, 1.0, 48000, 1, b"\x00\x00" * 100)
    await sink.write_chunk(chunk)
    await sink.stop()
    assert not out_wav.exists()


@pytest.mark.asyncio
async def test_ndi_audio_sink_mocked():
    with patch("dubber.sink.ndi_sink.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_send = MagicMock()
        mock_ndi.send_create.return_value = mock_send

        sink = NDIAudioSink(sink_name="Dub_EN", sample_rate=48000)
        await sink.start()

        pcm_data = b"\x00\x00" * 960
        chunk = BroadcastAudioChunk("sess", "en", "soniox", 0, 10.0, 48000, 1, pcm_data)
        await sink.write_chunk(chunk)

        assert mock_ndi.send_send_audio_v3.called
        assert mock_ndi.send_send_audio_v3.call_count == 1

        args, _ = mock_ndi.send_send_audio_v3.call_args
        assert args[0] == mock_send
        frame = args[1]
        assert frame.sample_rate == 48000
        assert frame.no_channels == 1
        assert frame.no_samples == 960  # 960 int16 samples (1920 bytes = 20 ms at 48 kHz)

        await sink.stop()
        mock_ndi.send_destroy.assert_called_once_with(mock_send)
        assert not hasattr(mock_ndi, "destroy") or not mock_ndi.destroy.called


@pytest.mark.asyncio
async def test_ndi_audio_sink_context_manager():
    with patch("dubber.sink.ndi_sink.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_send = MagicMock()
        mock_ndi.send_create.return_value = mock_send

        async with NDIAudioSink(sink_name="Dub_EN", sample_rate=48000) as sink:
            assert isinstance(sink, AudioSink)
            pcm_data = b"\x00\x00" * 960
            chunk = BroadcastAudioChunk("sess", "en", "gemini", 0, 10.0, 48000, 1, pcm_data)
            await sink.write_chunk(chunk)
            assert mock_ndi.send_send_audio_v3.called

        mock_ndi.send_destroy.assert_called_once_with(mock_send)


@pytest.mark.asyncio
async def test_ndi_audio_sink_init_failures():
    # 1. NDI not installed
    with patch("dubber.sink.ndi_sink.ndi", None):
        sink = NDIAudioSink()
        with pytest.raises(RuntimeError, match="NDIlib is not installed"):
            await sink.start()

    # 2. initialize() returns False
    with patch("dubber.sink.ndi_sink.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = False
        sink = NDIAudioSink()
        with pytest.raises(RuntimeError, match="Failed to initialize NDIlib"):
            await sink.start()

    # 3. send_create returns None
    with patch("dubber.sink.ndi_sink.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_ndi.send_create.return_value = None
        sink = NDIAudioSink(sink_name="Dub_Fail")
        with pytest.raises(RuntimeError, match="Failed to create NDI sender 'Dub_Fail'"):
            await sink.start()


@pytest.mark.asyncio
async def test_ndi_audio_sink_noop_when_not_started_or_empty():
    with patch("dubber.sink.ndi_sink.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        sink = NDIAudioSink()

        # Write chunk before start
        chunk = BroadcastAudioChunk("sess", "en", "gemini", 0, 10.0, 48000, 1, b"\x00\x00" * 960)
        await sink.write_chunk(chunk)
        assert not mock_ndi.send_send_audio_v3.called

        # Start and write empty chunk
        mock_send = MagicMock()
        mock_ndi.send_create.return_value = mock_send
        await sink.start()

        empty_chunk = BroadcastAudioChunk("sess", "en", "gemini", 1, 10.02, 48000, 1, b"")
        await sink.write_chunk(empty_chunk)
        assert not mock_ndi.send_send_audio_v3.called

        await sink.stop()


def test_sink_package_exports():
    from dubber.sink import AudioSink as AS
    from dubber.sink import NDIAudioSink as NS
    from dubber.sink import WAVAudioSink as WS

    assert AS is AudioSink
    assert NS is NDIAudioSink
    assert WS is WAVAudioSink


@pytest.mark.asyncio
async def test_ndi_audio_sink_real():
    try:
        import NDIlib as ndi
    except ImportError:
        ndi = None

    if ndi is None:
        pytest.skip("NDIlib is not installed")

    sink = NDIAudioSink(sink_name="Dub_EN_Test_Real", sample_rate=48000)
    await sink.start()

    pcm_data = (np.ones(960, dtype=np.int16) * 500).tobytes()
    chunk = BroadcastAudioChunk("sess_real", "en", "gemini", 0, 1.0, 48000, 1, pcm_data)
    await sink.write_chunk(chunk)
    await sink.stop()

