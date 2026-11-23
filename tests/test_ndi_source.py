"""Unit tests for NDIAudioSource ingest component."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dubber.ingest.ndi_source import NDIAudioSource
from dubber.types import AudioBlock


@pytest.mark.asyncio
async def test_ndi_audio_source_conversion():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "TEST_FEED (vMix)"
        mock_ndi.find_get_current_sources.return_value = [mock_src]

        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        # Mock frame capturing: 48 kHz stereo float32 frame of 960 samples (20 ms)
        audio_frame = MagicMock()
        audio_frame.sample_rate = 48000
        audio_frame.no_channels = 2
        audio_frame.no_samples = 960

        # Float32 stereo audio in [-1.0, 1.0]
        data = np.zeros((960, 2), dtype=np.float32)
        audio_frame.data = data
        mock_ndi.AudioFrameV3.return_value = audio_frame

        call_count = 0

        def capture_side_effect(recv, a_frame, v_frame, m_frame, timeout_ms):
            nonlocal call_count
            if call_count < 2:
                call_count += 1
                return mock_ndi.FRAME_TYPE_AUDIO
            return mock_ndi.FRAME_TYPE_NONE

        mock_ndi.recv_capture_v3.side_effect = capture_side_effect
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0

        source = NDIAudioSource(source_name="TEST_FEED", session_id="ndi_test")
        await source.start()

        block = await source.read_block()
        assert block is not None
        assert isinstance(block, AudioBlock)
        assert len(block.pcm_data) == 640  # 320 samples * 2 bytes = 20ms at 16 kHz
        assert block.session_id == "ndi_test"
        assert block.block_index == 0
        assert block.sample_offset == 0
        assert block.duration_ms == 20.0

        await source.stop()


@pytest.mark.asyncio
async def test_ndi_audio_source_int16_conversion():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "INT16_FEED"
        mock_ndi.find_get_current_sources.return_value = [mock_src]

        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        # Mock 48 kHz stereo int16 frame
        audio_frame = MagicMock()
        audio_frame.sample_rate = 48000
        audio_frame.no_channels = 2
        audio_frame.no_samples = 960
        audio_frame.data = np.ones((960, 2), dtype=np.int16) * 1000
        mock_ndi.AudioFrameV3.return_value = audio_frame

        mock_ndi.recv_capture_v3.return_value = mock_ndi.FRAME_TYPE_AUDIO

        source = NDIAudioSource(source_name="INT16_FEED", session_id="test_int16")
        await source.start()

        block = await source.read_block()
        assert block is not None
        assert len(block.pcm_data) == 640
        # Verify samples are converted properly
        samples = np.frombuffer(block.pcm_data, dtype=np.int16)
        assert len(samples) == 320
        assert np.allclose(samples[10:-10], 1000, atol=2)

        await source.stop()


@pytest.mark.asyncio
async def test_ndi_audio_source_first_available():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src1 = MagicMock()
        mock_src1.ndi_name = "DEFAULT_SOURCE"
        mock_src2 = MagicMock()
        mock_src2.ndi_name = "OTHER_SOURCE"
        mock_ndi.find_get_current_sources.return_value = [mock_src1, mock_src2]

        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        source = NDIAudioSource(source_name=None, session_id="first_avail")
        await source.start()

        assert source._running is True
        call_args = mock_ndi.recv_create_v3.call_args[0][0]
        assert call_args.source_to_connect_to == mock_src1

        await source.stop()
        assert source._running is False


@pytest.mark.asyncio
async def test_ndi_source_not_found():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_ndi.find_get_current_sources.return_value = []

        source = NDIAudioSource(source_name="NON_EXISTENT")
        with pytest.raises(RuntimeError, match="NDI source 'NON_EXISTENT' not found"):
            await source.start()


@pytest.mark.asyncio
async def test_ndi_not_installed():
    with patch("dubber.ingest.ndi_source.ndi", None):
        source = NDIAudioSource()
        with pytest.raises(RuntimeError, match="NDIlib is not installed"):
            await source.start()


@pytest.mark.asyncio
async def test_ndi_initialize_failed():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = False
        source = NDIAudioSource()
        with pytest.raises(RuntimeError, match="Failed to initialize NDIlib"):
            await source.start()


@pytest.mark.asyncio
async def test_ndi_free_audio_called():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "FREE_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]

        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        audio_frame = MagicMock()
        audio_frame.sample_rate = 16000
        audio_frame.no_channels = 1
        audio_frame.no_samples = 320
        audio_frame.data = np.zeros(320, dtype=np.int16)
        mock_ndi.AudioFrameV3.return_value = audio_frame

        mock_ndi.recv_capture_v3.return_value = mock_ndi.FRAME_TYPE_AUDIO

        source = NDIAudioSource(source_name="FREE_TEST")
        await source.start()
        block = await source.read_block()
        assert block is not None
        mock_ndi.recv_free_audio_v3.assert_called_once_with(mock_recv, audio_frame)
        await source.stop()


@pytest.mark.asyncio
async def test_native_pybind11_capture_fallback():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "PYBIND_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]

        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        audio_frame = MagicMock()
        audio_frame.sample_rate = 16000
        audio_frame.no_channels = 1
        audio_frame.no_samples = 320
        audio_frame.data = np.zeros(320, dtype=np.int16)

        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0

        # Simulate pybind11 signature: 5 args raises TypeError, (instance, timeout, v, a, m) succeeds
        def recv_capture_mock(*args, **kwargs):
            if len(args) == 5 and args[1] is None:
                raise TypeError("Incompatible function arguments")
            return (mock_ndi.FRAME_TYPE_AUDIO, None, audio_frame, None)

        mock_ndi.recv_capture_v3.side_effect = recv_capture_mock

        source = NDIAudioSource(source_name="PYBIND_TEST")
        async with source:
            block = await source.read_block()
            assert block is not None
            assert len(block.pcm_data) == 640


@pytest.mark.asyncio
async def test_ndi_multi_frame_buffering():
    """Verify that multiple smaller chunks (e.g. 10ms = 160 samples) are correctly buffered into 20ms blocks."""
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "BUFFER_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        frame_count = 0

        def capture_small_frames(*args, **kwargs):
            nonlocal frame_count
            f = MagicMock()
            f.sample_rate = 16000
            f.no_channels = 1
            f.no_samples = 160  # 10 ms chunks
            f.data = np.full(160, fill_value=frame_count, dtype=np.int16)
            mock_ndi.AudioFrameV3.return_value = f
            frame_count += 1
            return mock_ndi.FRAME_TYPE_AUDIO

        mock_ndi.recv_capture_v3.side_effect = capture_small_frames

        source = NDIAudioSource(source_name="BUFFER_TEST", session_id="buffering_test")
        await source.start()

        # Should require 2 captures of 160 samples to produce 1 block of 320 samples (640 bytes)
        block1 = await source.read_block()
        assert block1 is not None
        assert len(block1.pcm_data) == 640
        assert block1.block_index == 0
        assert block1.sample_offset == 0

        # Read second block
        block2 = await source.read_block()
        assert block2 is not None
        assert len(block2.pcm_data) == 640
        assert block2.block_index == 1
        assert block2.sample_offset == 320

        await source.stop()


@pytest.mark.asyncio
async def test_ndi_read_block_stopped():
    source = NDIAudioSource(source_name="STOPPED_TEST")
    # Not started yet
    assert await source.read_block() is None


@pytest.mark.asyncio
async def test_ndi_recv_create_failure():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "FAIL_SOURCE"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_ndi.recv_create_v3.return_value = None

        source = NDIAudioSource(source_name="FAIL_SOURCE")
        with pytest.raises(RuntimeError, match="Failed to connect to NDI source"):
            await source.start()


@pytest.mark.asyncio
async def test_ndi_find_create_failure():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_ndi.find_create_v2.return_value = None

        source = NDIAudioSource()
        with pytest.raises(RuntimeError, match="Failed to create NDI find instance"):
            await source.start()


@pytest.mark.asyncio
async def test_ndi_empty_frames_ignored():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "EMPTY_FRAME_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        call_idx = 0

        def capture_empty_then_valid(*args, **kwargs):
            nonlocal call_idx
            f = MagicMock()
            if call_idx == 0:
                # Empty frame (0 samples)
                f.sample_rate = 16000
                f.no_channels = 1
                f.no_samples = 0
                f.data = np.array([], dtype=np.int16)
            else:
                # Valid 320-sample frame
                f.sample_rate = 16000
                f.no_channels = 1
                f.no_samples = 320
                f.data = np.zeros(320, dtype=np.int16)
            mock_ndi.AudioFrameV3.return_value = f
            call_idx += 1
            return mock_ndi.FRAME_TYPE_AUDIO

        mock_ndi.recv_capture_v3.side_effect = capture_empty_then_valid

        source = NDIAudioSource(source_name="EMPTY_FRAME_TEST")
        await source.start()
        block = await source.read_block()
        assert block is not None
        assert len(block.pcm_data) == 640
        await source.stop()


@pytest.mark.asyncio
async def test_ndi_reset_state_on_start():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "RESET_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        source = NDIAudioSource(source_name="RESET_TEST")
        # Manually dirty internal state
        source._block_index = 42
        source._sample_offset = 13440
        source._pcm_buffer.extend(b"\x00" * 320)

        await source.start()
        assert source._block_index == 0
        assert source._sample_offset == 0
        assert len(source._pcm_buffer) == 0
        await source.stop()


@pytest.mark.asyncio
async def test_ndi_stop_does_not_destroy_global_ndi():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "GLOBAL_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        source = NDIAudioSource(source_name="GLOBAL_TEST")
        await source.start()
        await source.stop()

        mock_ndi.recv_destroy.assert_called_once_with(mock_recv)
        # Verify ndi.destroy is NOT called
        mock_ndi.destroy.assert_not_called()


@pytest.mark.asyncio
async def test_ndi_capture_mode_cached():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "CACHE_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        audio_frame = MagicMock()
        audio_frame.sample_rate = 16000
        audio_frame.no_channels = 1
        audio_frame.no_samples = 320
        audio_frame.data = np.zeros(320, dtype=np.int16)

        call_log = []

        def recv_capture(*args, **kwargs):
            call_log.append(args)
            if len(args) == 5 and args[1] is None:
                raise TypeError("Incompatible function arguments")
            return (mock_ndi.FRAME_TYPE_AUDIO, None, audio_frame, None)

        mock_ndi.recv_capture_v3.side_effect = recv_capture

        source = NDIAudioSource(source_name="CACHE_TEST")
        await source.start()
        assert source._capture_mode is None

        # First read triggers TypeError and caches 'pybind11'
        block1 = await source.read_block()
        assert block1 is not None
        assert source._capture_mode == "pybind11"

        # Second read uses 'pybind11' directly without TypeError
        block2 = await source.read_block()
        assert block2 is not None
        # In call_log, after the first 5-arg failed attempt, all calls have len 5 with args[1]==50 (pybind11 call)
        pybind11_calls = [c for c in call_log if c[1] == 50]
        assert len(pybind11_calls) == 2

        await source.stop()


@pytest.mark.asyncio
async def test_ndi_free_frame_on_processing_error():
    with patch("dubber.ingest.ndi_source.ndi") as mock_ndi:
        mock_ndi.FRAME_TYPE_AUDIO = 2
        mock_ndi.FRAME_TYPE_NONE = 0
        mock_ndi.initialize.return_value = True
        mock_find = MagicMock()
        mock_ndi.find_create_v2.return_value = mock_find
        mock_src = MagicMock()
        mock_src.ndi_name = "FREE_GUARD_TEST"
        mock_ndi.find_get_current_sources.return_value = [mock_src]
        mock_recv = MagicMock()
        mock_ndi.recv_create_v3.return_value = mock_recv

        audio_frame = MagicMock()
        mock_ndi.AudioFrameV3.return_value = audio_frame
        mock_ndi.recv_capture_v3.return_value = mock_ndi.FRAME_TYPE_AUDIO

        source = NDIAudioSource(source_name="FREE_GUARD_TEST")
        await source.start()

        # Simulate exception during frame processing
        with (
            patch.object(source, "_process_frame_to_pcm16_16k", side_effect=ValueError("DSP Error")),
            pytest.raises(ValueError, match="DSP Error"),
        ):
            await source.read_block()

        # Verify free frame was still called via finally
        mock_ndi.recv_free_audio_v3.assert_called_once_with(mock_recv, audio_frame)
        await source.stop()


