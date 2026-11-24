import asyncio
import json
import urllib.request
import wave

import numpy as np
import pytest

from dubber.cli import build_parser, main, run_operator_service, run_test_pipeline
from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.ingest.wav_source import WAVAudioSource
from dubber.measure.latency import LatencyAnalyzer
from dubber.measure.report import generate_latency_report
from dubber.sink.wav_sink import WAVAudioSink


def test_latency_analyzer_and_report():
    analyzer = LatencyAnalyzer(max_allowed_s=3.0)
    r1 = analyzer.compute_latency("phrase_1", 1.0, 2.5)  # 1.5s -> pass
    r2 = analyzer.compute_latency("phrase_2", 4.0, 7.2)  # 3.2s -> fail
    assert r1.passed is True
    assert r2.passed is False

    report = generate_latency_report([r1, r2])
    assert "PHARMALIVE DUBBER LATENCY REPORT" in report
    assert "Median latency" in report
    assert "phrase_1" in report
    assert "phrase_2" in report


def test_detect_speech_start():
    analyzer = LatencyAnalyzer(energy_threshold=500.0)
    sr = 48000
    # 0.5s silence followed by 0.5s 440Hz tone at 48000 Hz
    silence = np.zeros(int(0.5 * sr), dtype=np.int16)
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)) * 32767).astype(np.int16)
    audio = np.concatenate([silence, tone])

    onset_s = analyzer.detect_speech_start(audio, sample_rate=sr, window_ms=20)
    assert 0.48 <= onset_s <= 0.52

    # Pure silence must return -1.0
    silent_audio = np.zeros(sr, dtype=np.int16)
    assert analyzer.detect_speech_start(silent_audio, sample_rate=sr) == -1.0


@pytest.mark.asyncio
async def test_end_to_end_pipeline(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    state_file = tmp_path / "state.json"

    # 0.5s tone audio
    samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000, endpoint=False)) * 32767).astype(np.int16)
    with wave.open(str(in_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())

    config = DubberConfig(
        source_type="wav",
        provider_mode="mock",
        wav_file_path=str(in_wav),
        state_file_path=str(state_file),
    )
    source = WAVAudioSource(str(in_wav), real_time_pace=False)
    sink = WAVAudioSink(str(out_wav), sample_rate=48000, channels=1)

    engine = DubberEngine(config, source=source, sink=sink)
    engine.gemini_worker.simulated_delay_s = 0.05
    engine.soniox_worker.simulated_stt_delay_s = 0.02
    engine.soniox_worker.simulated_tts_delay_s = 0.02

    await engine.start()
    # In non-real-time mode, source will empty quickly
    while source._running:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)
    await engine.stop()

    assert out_wav.exists()
    assert out_wav.stat().st_size > 0


@pytest.mark.asyncio
async def test_cli_test_run(tmp_path, capsys):
    in_wav = tmp_path / "test_ru.wav"
    out_wav = tmp_path / "out_en.wav"
    state_file = tmp_path / "state.json"

    # Generate 0.4s tone
    samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.4, 6400, endpoint=False)) * 32767).astype(np.int16)
    with wave.open(str(in_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())

    exit_code = await run_test_pipeline(
        wav_input=str(in_wav),
        wav_output=str(out_wav),
        state_file=str(state_file),
        active_branch="gemini",
        max_allowed_s=3.0,
    )
    assert exit_code == 0
    assert out_wav.exists()

    captured = capsys.readouterr().out
    assert "PHARMALIVE DUBBER LATENCY REPORT" in captured


def test_empty_latency_report():
    report = generate_latency_report([])
    assert "PHARMALIVE DUBBER LATENCY REPORT" in report
    assert "No latency measurements available" in report


def test_main_cli_dispatch(tmp_path, capsys):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    state_file = tmp_path / "state.json"

    # Test main without command
    code_no_cmd = main([])
    assert code_no_cmd == 1

    # Test main with test-run using --wav and --out-wav flags
    samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, 3200, endpoint=False)) * 32767).astype(np.int16)
    with wave.open(str(in_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())

    code = main([
        "test-run",
        "--wav", str(in_wav),
        "--out-wav", str(out_wav),
        "--state-file", str(state_file),
        "--active-branch", "soniox",
        "--threshold", "3.0",
        "--simulated-delay", "0.02",
    ])
    assert code == 0
    assert out_wav.exists()
    out = capsys.readouterr().out
    assert "PHARMALIVE DUBBER LATENCY REPORT" in out


def test_cli_flags_parsing():
    parser = build_parser()

    # Verify run flags
    run_args = parser.parse_args([
        "run",
        "--sink", "wav",
        "--wav", "data/custom_ru.wav",
        "--out-wav", "output/custom_out.wav",
        "--no-tui",
    ])
    assert run_args.sink == "wav"
    assert run_args.wav == "data/custom_ru.wav"
    assert run_args.out_wav == "output/custom_out.wav"
    assert run_args.no_tui is True

    # Verify run flags alias --wav-file
    run_alias_args = parser.parse_args([
        "run",
        "--wav-file", "data/alias_ru.wav",
    ])
    assert run_alias_args.wav == "data/alias_ru.wav"

    # Verify test-run flags: --wav and --out-wav
    test_args = parser.parse_args([
        "test-run",
        "--wav", "data/test_in.wav",
        "--out-wav", "output/test_out.wav",
    ])
    assert test_args.wav == "data/test_in.wav"
    assert test_args.out_wav == "output/test_out.wav"

    # Verify test-run aliases: --input and --output
    test_alias_args = parser.parse_args([
        "test-run",
        "-i", "data/input.wav",
        "-o", "output/output.wav",
    ])
    assert test_alias_args.wav == "data/input.wav"
    assert test_alias_args.out_wav == "output/output.wav"


@pytest.mark.asyncio
async def test_run_operator_service_headless(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    state_file = tmp_path / "state.json"
    samples = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, 3200, endpoint=False)) * 32767).astype(np.int16)
    with wave.open(str(in_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())

    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--source", "wav",
        "--sink", "wav",
        "--wav", str(in_wav),
        "--out-wav", str(out_wav),
        "--state-file", str(state_file),
        "--port", "8999",
        "--host", "127.0.0.1",
        "--no-tui",
    ])

    service_task = asyncio.create_task(run_operator_service(args))
    # Give server time to bind
    await asyncio.sleep(0.5)

    try:
        loop = asyncio.get_running_loop()

        def fetch():
            with urllib.request.urlopen("http://127.0.0.1:8999/status", timeout=2.0) as req:
                return req.status, json.loads(req.read().decode("utf-8"))

        status_code, data = await loop.run_in_executor(None, fetch)
        assert status_code == 200
        assert "active_branch" in data
        assert data["active_branch"] == "gemini"
    finally:
        service_task.cancel()
        try:
            await service_task
        except asyncio.CancelledError:
            pass
