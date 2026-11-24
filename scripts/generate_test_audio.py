"""Synthetic speech WAV audio generator for PharmaLive Dubber acceptance testing."""

import argparse
import os
import sys
import wave

import numpy as np


def generate_speech_harmonics(
    duration_s: float,
    sample_rate: int = 16000,
    f0: float = 160.0,
    syllable_rate: float = 3.5,
) -> np.ndarray:
    """Synthesize speech-like harmonic vowel tones with syllable envelope modulation."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)

    # Multi-harmonic stack simulating voiced speech sounds / formants
    harmonics = [
        (f0, 0.40),
        (f0 * 2, 0.25),
        (f0 * 3, 0.15),
        (f0 * 4, 0.10),
        (f0 * 5, 0.05),
    ]

    tone = np.zeros(num_samples, dtype=np.float64)
    for freq, weight in harmonics:
        if freq < sample_rate / 2:
            tone += weight * np.sin(2 * np.pi * freq * t)

    # Syllable modulation (~3-4 Hz speech envelope cadence)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t - np.pi / 2))
    modulated = tone * (0.35 + 0.65 * syllable_env)

    # Smooth attack and decay ramp (20 ms cosine taper) to eliminate onset/offset clicks
    fade_len = min(int(0.02 * sample_rate), num_samples // 4)
    if fade_len > 0:
        ramp_up = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade_len)))
        ramp_down = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_len)))
        modulated[:fade_len] *= ramp_up
        modulated[-fade_len:] *= ramp_down

    # Normalize to 70% full scale int16 range
    max_val = np.max(np.abs(modulated))
    if max_val > 0:
        normalized = (modulated / max_val) * 0.70 * 32767
    else:
        normalized = np.zeros(num_samples)

    return normalized.astype(np.int16)


def generate_test_speech(
    file_path: str = "data/test_speech_ru.wav",
    duration_s: float = 6.0,
    sample_rate: int = 16000,
) -> str:
    """Generate synthetic Russian test speech WAV file (16 kHz mono 16-bit PCM).

    Structure:
    - Initial silence:      0.0s - 0.5s
    - Phrase 1 (speech):   0.5s - 2.5s (2.0s duration, ~160 Hz fundamental + harmonics)
    - Inter-phrase pause:   2.5s - 3.5s (1.0s duration, silence)
    - Phrase 2 (speech):   3.5s - 5.5s (2.0s duration, ~200 Hz fundamental + harmonics)
    - Trailing silence:     5.5s - 6.0s (0.5s duration, silence)
    """
    total_samples = int(duration_s * sample_rate)
    audio = np.zeros(total_samples, dtype=np.int16)

    # Section timing indices
    p1_start = int(0.5 * sample_rate)
    p1_end = int(2.5 * sample_rate)
    p2_start = int(3.5 * sample_rate)
    p2_end = int(5.5 * sample_rate)

    # Phrase 1: 0.5s - 2.5s (duration 2.0s)
    p1_samples = generate_speech_harmonics(
        duration_s=(p1_end - p1_start) / sample_rate,
        sample_rate=sample_rate,
        f0=160.0,
        syllable_rate=3.5,
    )
    audio[p1_start:p1_end] = p1_samples

    # Inter-phrase pause (2.5s - 3.5s) remains zeros

    # Phrase 2: 3.5s - 5.5s (duration 2.0s)
    p2_samples = generate_speech_harmonics(
        duration_s=(p2_end - p2_start) / sample_rate,
        sample_rate=sample_rate,
        f0=200.0,
        syllable_rate=4.0,
    )
    audio[p2_start:p2_end] = p2_samples

    # Trailing silence (5.5s - 6.0s) remains zeros

    # Ensure parent directory exists
    dir_path = os.path.dirname(os.path.abspath(file_path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)  # 16 kHz
        wf.writeframes(audio.tobytes())

    return file_path


def main(argv: list[str] | None = None) -> int:
    """CLI parser and entry point for test audio generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Russian test speech WAV file for PharmaLive Dubber"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/test_speech_ru.wav",
        help="Target output WAV path (default: data/test_speech_ru.wav)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=6.0,
        help="Total duration in seconds (default: 6.0)",
    )
    args = parser.parse_args(argv)

    out_path = generate_test_speech(file_path=args.output, duration_s=args.duration)
    print(f"Generated synthetic speech asset: {out_path} ({args.duration:.1f}s, 16 kHz mono 16-bit PCM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
