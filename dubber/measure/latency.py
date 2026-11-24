"""End-to-end latency measurement and audio speech onset detection."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyResult:
    """Result of end-to-end latency evaluation for a speech phrase or block."""

    phrase_id: str
    t_in: float
    t_out: float
    latency_s: float
    passed: bool


def detect_speech_start(
    audio_data: np.ndarray,
    sample_rate: int = 48000,
    window_ms: int = 20,
    threshold: float = 500.0,
) -> float:
    """Detect speech onset time (in seconds) via energy threshold / VAD.

    Args:
        audio_data: 1D or 2D numpy array containing PCM audio samples (int16 or float).
        sample_rate: Sampling rate of audio_data in Hz (default: 48000).
        window_ms: Window analysis duration in ms (default: 20 ms).
        threshold: Energy / RMS amplitude threshold for detecting speech (default: 500.0).

    Returns:
        Time offset in seconds where speech energy is first detected, or -1.0 if not detected (silence).
    """
    if audio_data is None or len(audio_data) == 0:
        return -1.0

    data = audio_data.astype(np.float32)

    # Downmix multi-channel to mono if necessary
    if data.ndim > 1:
        data = np.mean(data, axis=1 if data.shape[1] < data.shape[0] else 0)

    # If normalized float in [-1.0, 1.0] but threshold is in int16 scale (> 1.0):
    if threshold > 1.0 and np.max(np.abs(data)) <= 1.0 and np.max(np.abs(data)) > 0:
        data = data * 32767.0

    window_samples = max(1, int(sample_rate * (window_ms / 1000.0)))
    num_windows = len(data) // window_samples
    if num_windows == 0:
        return -1.0

    truncated = data[: num_windows * window_samples]
    windows = truncated.reshape((num_windows, window_samples))
    rms = np.sqrt(np.mean(windows**2, axis=1))

    # Detect speech onset
    active_windows = np.where(rms >= threshold)[0]
    if len(active_windows) == 0:
        return -1.0

    start_window = int(active_windows[0])
    return float((start_window * window_samples) / sample_rate)


class LatencyAnalyzer:
    """Analyzes end-to-end latency and speech onset for live dubbing."""

    def __init__(
        self,
        energy_threshold: float = 500.0,
        max_allowed_s: float = 3.0,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.max_allowed_s = max_allowed_s

    def compute_latency(
        self, phrase_id: str, t_in: float, t_out: float
    ) -> LatencyResult:
        """Compute latency between source speech onset and translated speech delivery."""
        latency_s = round(max(0.0, t_out - t_in), 4)
        passed = latency_s <= self.max_allowed_s
        return LatencyResult(
            phrase_id=phrase_id,
            t_in=round(t_in, 4),
            t_out=round(t_out, 4),
            latency_s=latency_s,
            passed=passed,
        )

    def detect_speech_start(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 48000,
        window_ms: int = 20,
        threshold: float | None = None,
    ) -> float:
        """Detect speech start time via VAD energy threshold using configured threshold."""
        thresh = self.energy_threshold if threshold is None else threshold
        return detect_speech_start(
            audio_data=audio_data,
            sample_rate=sample_rate,
            window_ms=window_ms,
            threshold=thresh,
        )
