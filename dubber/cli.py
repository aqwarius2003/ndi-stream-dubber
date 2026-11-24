"""Command-line interface (CLI) for PharmaLive Dubber."""

import argparse
import asyncio
import logging
import os
import sys
import time
import wave
from typing import Any

import numpy as np
import uvicorn

from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.ingest.wav_source import WAVAudioSource
from dubber.measure.latency import LatencyAnalyzer, LatencyResult
from dubber.measure.report import generate_latency_report
from dubber.operator.server import create_app
from dubber.sink.wav_sink import WAVAudioSink
from dubber.types import BroadcastAudioChunk

logger = logging.getLogger(__name__)


def _generate_sample_wav(
    file_path: str, duration_s: float = 1.0, sample_rate: int = 16000
) -> None:
    """Generate a clean test sine wave PCM WAV file if the input does not exist."""
    dirname = os.path.dirname(os.path.abspath(file_path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class _MeasuringWAVAudioSink(WAVAudioSink):
    """WAV sink that measures latency for each written BroadcastAudioChunk."""

    def __init__(
        self,
        file_path: str,
        analyzer: LatencyAnalyzer,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        super().__init__(file_path, sample_rate=sample_rate, channels=channels)
        self.analyzer = analyzer
        self.results: list[LatencyResult] = []

    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        if chunk.pcm_data:
            t_out = time.monotonic()
            t_in = chunk.source_time
            phrase_id = f"block_{chunk.block_index:04d}"
            res = self.analyzer.compute_latency(phrase_id, t_in, t_out)
            self.results.append(res)
        await super().write_chunk(chunk)


async def run_test_pipeline(
    wav_input: str = "data/test_speech_ru.wav",
    wav_output: str = "output/test_broadcast_en.wav",
    state_file: str = "state.json",
    active_branch: str = "gemini",
    mode: str = "mock",
    max_allowed_s: float = 3.0,
    simulated_delay_s: float = 0.05,
    real_time_pace: bool = False,
) -> int:
    """Execute reproducible WAV test run, measure latency, and print SLA report."""
    if not os.path.exists(wav_input):
        logger.info("Input WAV '%s' not found. Creating sample tone audio.", wav_input)
        _generate_sample_wav(wav_input, duration_s=1.0)

    config = DubberConfig.from_env()
    config.source_type = "wav"
    config.provider_mode = mode  # type: ignore[arg-type]
    config.wav_file_path = wav_input
    config.state_file_path = state_file

    analyzer = LatencyAnalyzer(max_allowed_s=max_allowed_s)
    source = WAVAudioSource(wav_input, real_time_pace=real_time_pace)
    sink = _MeasuringWAVAudioSink(
        wav_output,
        analyzer=analyzer,
        sample_rate=48000,
        channels=1,
    )

    engine = DubberEngine(config, source=source, sink=sink)
    if mode == "mock":
        engine.gemini_worker.simulated_delay_s = simulated_delay_s
        engine.soniox_worker.simulated_stt_delay_s = simulated_delay_s / 2
        engine.soniox_worker.simulated_tts_delay_s = simulated_delay_s / 2

    engine.selector.switch_branch(active_branch, force=True)  # type: ignore[arg-type]

    await engine.start()
    try:
        while source._running:
            await asyncio.sleep(0.05)
        # Allow worker processing queues to flush
        await asyncio.sleep(simulated_delay_s * 2 + 0.2)
    finally:
        await engine.stop()

    report = generate_latency_report(sink.results, threshold_s=max_allowed_s)
    print(report)

    all_passed = all(r.passed for r in sink.results)
    return 0 if (all_passed and len(sink.results) > 0) or len(sink.results) == 0 else 1


async def run_operator_service(args: argparse.Namespace) -> int:
    """Launch engine, background FastAPI control server, and Rich TUI dashboard."""
    config = DubberConfig.from_env()
    if getattr(args, "source", None):
        config.source_type = args.source
    if getattr(args, "mode", None):
        config.provider_mode = args.mode
    if getattr(args, "wav", None):
        config.wav_file_path = args.wav
    if getattr(args, "ndi_source", None):
        config.ndi_source_name = args.ndi_source
    if getattr(args, "ndi_sink", None):
        config.ndi_sink_name = args.ndi_sink
    if getattr(args, "state_file", None):
        config.state_file_path = args.state_file
    if getattr(args, "port", None):
        config.http_port = args.port
    if getattr(args, "host", None):
        config.http_host = args.host

    if config.source_type == "wav":
        wav_path = config.wav_file_path or "data/test_speech_ru.wav"
        if not os.path.exists(wav_path):
            _generate_sample_wav(wav_path)
        source = WAVAudioSource(wav_path, real_time_pace=True)
    else:
        from dubber.ingest.ndi_source import NDIAudioSource

        source = NDIAudioSource(source_name=config.ndi_source_name)

    sink_type = getattr(args, "sink", "ndi")
    out_wav_path = getattr(args, "out_wav", "output/broadcast_en.wav")

    sink: Any = None
    if sink_type == "wav":
        sink = WAVAudioSink(out_wav_path)
    else:
        try:
            from dubber.sink.ndi_sink import NDIAudioSink

            sink = NDIAudioSink(sink_name=config.ndi_sink_name)
        except (RuntimeError, ImportError, OSError) as exc:
            logger.warning(
                "Could not initialize NDIAudioSink (%s). Falling back to WAVAudioSink('%s').",
                exc,
                out_wav_path,
            )
            sink = WAVAudioSink(out_wav_path)

    engine = DubberEngine(config, source=source, sink=sink)
    active_branch = getattr(args, "active_branch", "gemini")
    if active_branch:
        engine.selector.switch_branch(active_branch, force=True)

    await engine.start()

    app = create_app(engine)
    server_config = uvicorn.Config(
        app=app,
        host=config.http_host,
        port=config.http_port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    server_task = asyncio.create_task(server.serve())

    stop_event = asyncio.Event()

    try:
        if getattr(args, "no_tui", False):
            logger.info("Dubber operator server running at http://%s:%d", config.http_host, config.http_port)
            print(f"PharmaLive Dubber running. Control API: http://{config.http_host}:{config.http_port}")
            print("Press Ctrl+C to terminate.")
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        else:
            from dubber.operator.cli import run_operator_tui

            await run_operator_tui(engine, stop_event=stop_event)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.should_exit = True
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        await engine.stop()

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with run and test-run subcommands."""
    parser = argparse.ArgumentParser(
        prog="dubber",
        description="PharmaLive Dubber: Real-time dual-branch RU->EN simultaneous dubbing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Launch live dubbing pipeline with API and TUI")
    run_parser.add_argument(
        "--source",
        choices=["wav", "ndi"],
        default="wav",
        help="Input audio source type (default: wav)",
    )
    run_parser.add_argument(
        "--sink",
        choices=["ndi", "wav"],
        default="ndi",
        help="Output audio sink publisher type (default: ndi)",
    )
    run_parser.add_argument(
        "--wav",
        "--wav-file",
        dest="wav",
        default="data/test_speech_ru.wav",
        help="Input WAV file path for wav source (default: data/test_speech_ru.wav)",
    )
    run_parser.add_argument(
        "--out-wav",
        dest="out_wav",
        default="output/broadcast_en.wav",
        help="Output WAV file path when using wav sink (default: output/broadcast_en.wav)",
    )
    run_parser.add_argument(
        "--ndi-source",
        default=None,
        help="NDI source name pattern to capture",
    )
    run_parser.add_argument(
        "--ndi-sink",
        default="Dub_EN",
        help="NDI publisher output stream name (default: Dub_EN)",
    )
    run_parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Translation worker provider mode (default: mock)",
    )
    run_parser.add_argument(
        "--active-branch",
        choices=["gemini", "soniox"],
        default="gemini",
        help="Initial active broadcast branch (default: gemini)",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP API control port (default: 8000)",
    )
    run_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP API bind address (default: 127.0.0.1)",
    )
    run_parser.add_argument(
        "--state-file",
        default="state.json",
        help="Persistent state file path (default: state.json)",
    )
    run_parser.add_argument(
        "--no-tui",
        action="store_true",
        default=False,
        help="Run without interactive Rich TUI console (headless mode)",
    )

    # 'test-run' subcommand
    test_parser = subparsers.add_parser(
        "test-run",
        help="Execute reproducible WAV test run, record latency, and print report",
    )
    test_parser.add_argument(
        "--wav",
        "--input",
        "-i",
        dest="wav",
        default="data/test_speech_ru.wav",
        help="Input WAV file path (default: data/test_speech_ru.wav)",
    )
    test_parser.add_argument(
        "--out-wav",
        "--output",
        "-o",
        dest="out_wav",
        default="output/test_broadcast_en.wav",
        help="Output WAV file path (default: output/test_broadcast_en.wav)",
    )
    test_parser.add_argument(
        "--state-file",
        default="state.json",
        help="Persistent state file path (default: state.json)",
    )
    test_parser.add_argument(
        "--active-branch",
        choices=["gemini", "soniox"],
        default="gemini",
        help="Active branch to evaluate (default: gemini)",
    )
    test_parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Provider mode (default: mock)",
    )
    test_parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Maximum allowed latency threshold in seconds (default: 3.0)",
    )
    test_parser.add_argument(
        "--simulated-delay",
        type=float,
        default=0.05,
        help="Simulated processing delay in mock mode (default: 0.05s)",
    )
    test_parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Pace input audio in real time (default: False for fast testing)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dubber command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        return asyncio.run(run_operator_service(args))
    elif args.command == "test-run":
        return asyncio.run(
            run_test_pipeline(
                wav_input=args.wav,
                wav_output=args.out_wav,
                state_file=args.state_file,
                active_branch=args.active_branch,
                mode=args.mode,
                max_allowed_s=args.threshold,
                simulated_delay_s=args.simulated_delay,
                real_time_pace=args.real_time,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
