import os
import wave

import numpy as np
import pytest

from scripts.generate_test_audio import (
    generate_speech_harmonics,
    generate_test_speech,
    main,
)


def test_generate_speech_harmonics_empty_duration():
    res_zero = generate_speech_harmonics(duration_s=0.0)
    assert isinstance(res_zero, np.ndarray)
    assert len(res_zero) == 0
    assert res_zero.dtype == np.int16

    res_neg = generate_speech_harmonics(duration_s=-1.0)
    assert len(res_neg) == 0
    assert res_neg.dtype == np.int16


def test_generate_test_speech_duration_validation():
    with pytest.raises(ValueError, match="duration_s must be >= 6.0s"):
        generate_test_speech(duration_s=4.0)


def test_generate_test_speech_creates_valid_wav(tmp_path):
    output_wav = str(tmp_path / "test_ru.wav")
    result_path = generate_test_speech(file_path=output_wav, duration_s=6.0)
    assert os.path.exists(output_wav)
    assert result_path == output_wav

    with wave.open(output_wav, "rb") as wf:
        assert wf.getnchannels() == 1  # mono
        assert wf.getsampwidth() == 2  # 16-bit PCM
        assert wf.getframerate() == 16000  # 16 kHz
        n_frames = wf.getnframes()
        assert n_frames == 16000 * 6  # 6.0 seconds = 96,000 samples
        frames = wf.readframes(n_frames)

    samples = np.frombuffer(frames, dtype=np.int16)
    sr = 16000

    # Section 1: Initial silence (0.0s - 0.5s)
    silence1 = samples[: int(0.5 * sr)]
    assert np.all(silence1 == 0)

    # Section 2: Phrase 1 (0.5s - 2.5s)
    phrase1 = samples[int(0.5 * sr) : int(2.5 * sr)]
    assert np.max(np.abs(phrase1)) > 1000
    rms_p1 = np.sqrt(np.mean(phrase1.astype(np.float64) ** 2))
    assert rms_p1 > 1000

    # Section 3: Inter-phrase pause (2.5s - 3.5s)
    pause = samples[int(2.5 * sr) : int(3.5 * sr)]
    assert np.all(pause == 0)

    # Section 4: Phrase 2 (3.5s - 5.5s)
    phrase2 = samples[int(3.5 * sr) : int(5.5 * sr)]
    assert np.max(np.abs(phrase2)) > 1000
    rms_p2 = np.sqrt(np.mean(phrase2.astype(np.float64) ** 2))
    assert rms_p2 > 1000

    # Section 5: Trailing silence (5.5s - 6.0s)
    silence2 = samples[int(5.5 * sr) :]
    assert np.all(silence2 == 0)


def test_generate_test_speech_cli_main(tmp_path, monkeypatch):
    out_file = str(tmp_path / "cli_speech.wav")
    monkeypatch.setattr("sys.argv", ["generate_test_audio.py", "--output", out_file])
    exit_code = main()
    assert exit_code == 0
    assert os.path.exists(out_file)


def test_generate_test_speech_cli_main_invalid_duration(tmp_path, monkeypatch, capsys):
    out_file = str(tmp_path / "invalid.wav")
    monkeypatch.setattr("sys.argv", ["generate_test_audio.py", "--output", out_file, "--duration", "3.0"])
    exit_code = main()
    assert exit_code == 1
    captured = capsys.readouterr().err
    assert "duration_s must be >= 6.0s" in captured
