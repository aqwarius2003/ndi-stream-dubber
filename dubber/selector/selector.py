"""DubSelector: operator-controlled broadcast audio branch selector.

Enforces zero-auto-fallback, target freshness gates, and crossfades on switches.
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np

from dubber.selector.state import SelectorState
from dubber.types import BranchState, BranchType, BroadcastAudioChunk

logger = logging.getLogger(__name__)


class DubSelector:
    """Manages audio source selection between translation branches (e.g. Gemini vs Soniox).

    Key architectural rules (SPEC Section 3, GEMINI.md):
    - Zero auto-fallback: Active branch failure triggers alert, outputs silence, but NEVER auto-switches.
    - Operator authority: Branch switching ONLY occurs upon explicit operator command.
    - Freshness check on switch: Target branch must be READY/RUNNING and fresh (<= 0.5s).
    - Crossfade: 15 ms crossfade (720 samples at 48 kHz) to prevent acoustic clicks on transition.
    - Persistent state: Remembers active selection across restarts in state.json.
    - Thread-safety: Thread-safe locks across operator controls and audio pipeline.
    """

    def __init__(
        self,
        state_file_path: str | Path | None = None,
        default_branch: BranchType = "gemini",
        freshness_threshold_s: float = 0.5,
        crossfade_ms: int = 15,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        self.state_file_path: Path | None = Path(state_file_path) if state_file_path else None
        self.default_branch: BranchType = default_branch
        self.freshness_threshold_s: float = freshness_threshold_s
        self.crossfade_ms: int = crossfade_ms
        self.sample_rate: int = sample_rate
        self.channels: int = channels

        # Thread safety lock
        self._lock = threading.RLock()

        # Crossfade duration in frames
        self.crossfade_samples: int = int(sample_rate * (crossfade_ms / 1000.0))

        # Persistent state
        if self.state_file_path is not None:
            self._state = SelectorState.load(self.state_file_path, default_branch=self.default_branch)
        else:
            self._state = SelectorState(active_branch=self.default_branch)

        # Monitored branch states
        self._branch_states: dict[BranchType, BranchState] = {
            "gemini": BranchState(branch="gemini", status="STARTING"),
            "soniox": BranchState(branch="soniox", status="STARTING"),
        }

        # Crossfade buffer state
        self._last_tail: np.ndarray | None = None
        self._fade_old_tail: np.ndarray | None = None
        self._fade_pos: int = 0
        self._is_crossfading: bool = False

    @property
    def active_branch(self) -> BranchType:
        """Currently active broadcast branch."""
        with self._lock:
            return self._state.active_branch

    def update_branch_state(self, state: BranchState) -> None:
        """Update monitored status of a translation branch."""
        with self._lock:
            self._branch_states[state.branch] = state

    def get_branch_state(self, branch: BranchType) -> BranchState:
        """Get the latest recorded state of a branch."""
        with self._lock:
            if branch not in self._branch_states:
                self._branch_states[branch] = BranchState(branch=branch, status="STARTING")
            return self._branch_states[branch]

    def is_alert_active(self) -> bool:
        """Return True if the currently active branch is in an error or failed state.

        Zero auto-fallback rule: selector flags alert, but NEVER auto-switches.
        """
        with self._lock:
            active_state = self.get_branch_state(self.active_branch)
            if active_state.status in ("ERROR", "STALE", "STOPPED"):
                return True
            return bool(active_state.error_message)

    def switch_branch(self, target: BranchType, force: bool = False) -> tuple[bool, str]:
        """Switch the active broadcast branch to target.

        Enforces freshness check:
        - Target must have status READY or RUNNING.
        - Target's last audio must not be older than freshness_threshold_s.

        Args:
            target: The target branch ('gemini' or 'soniox').
            force: If True, bypass readiness and freshness checks.

        Returns:
            tuple[bool, str]: (Success flag, descriptive message).
        """
        with self._lock:
            if target not in ("gemini", "soniox"):
                return False, f"Unknown branch: '{target}'"

            old_branch = self._state.active_branch
            if target == old_branch:
                return True, f"Branch '{target}' is already active"

            target_state = self.get_branch_state(target)

            if not force:
                if target_state.status not in ("READY", "RUNNING"):
                    msg = f"Target branch '{target}' is NOT READY (status: {target_state.status})"
                    logger.warning("Rejected switch command: %s", msg)
                    return False, msg

                now = time.monotonic()
                age = now - target_state.last_audio_time
                if target_state.last_audio_time <= 0.0 or age > self.freshness_threshold_s:
                    msg = (
                        f"Target branch '{target}' is NOT FRESH "
                        f"(age: {age:.2f}s > {self.freshness_threshold_s:.2f}s)"
                    )
                    logger.warning("Rejected switch command: %s", msg)
                    return False, msg

            self._state.active_branch = target
            self._state.last_switch_time = time.time()
            self._state.switch_count += 1
            self._save_state()
            self._trigger_crossfade()
            logger.info("Switched active branch from %s to %s", old_branch, target)
            return True, f"Switched active branch from '{old_branch}' to '{target}'"

    def process_chunk(self, chunk: BroadcastAudioChunk) -> BroadcastAudioChunk | None:
        """Process an incoming audio chunk from a translation branch.

        - Updates incoming branch timestamp and promotes STARTING to READY.
        - Discards chunks from non-active branches (returns None).
        - If active branch is failed (is_alert_active()), outputs silence.
        - Applies crossfade smoothing when switching between branches.
        """
        with self._lock:
            # Update incoming branch timestamp and status before filtering
            src = chunk.source_branch
            if src in self._branch_states:
                b_state = self._branch_states[src]
                b_state.last_audio_time = time.monotonic()
                if b_state.status == "STARTING":
                    b_state.status = "READY"
            elif src in ("gemini", "soniox"):
                self._branch_states[src] = BranchState(
                    branch=src,
                    status="READY",
                    last_audio_time=time.monotonic(),
                )

            # Discard audio from inactive branches
            if chunk.source_branch != self.active_branch:
                return None

            # Align byte buffer to 16-bit multi-channel frame boundary
            align_bytes = 2 * max(1, self.channels)
            pcm_bytes = chunk.pcm_data
            if len(pcm_bytes) % align_bytes != 0:
                pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % align_bytes)]

            # Zero auto-fallback rule: active failure outputs silence
            if self.is_alert_active():
                silence_pcm = b"\x00" * len(pcm_bytes)
                total_samples = len(pcm_bytes) // 2
                if total_samples > 0:
                    silence_len = min(self.crossfade_samples * self.channels, total_samples)
                    self._last_tail = np.zeros(silence_len, dtype=np.int16)
                return BroadcastAudioChunk(
                    session_id=chunk.session_id,
                    language=chunk.language,
                    source_branch=chunk.source_branch,
                    block_index=chunk.block_index,
                    source_time=chunk.source_time,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    pcm_data=silence_pcm,
                )

            samples = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
            total_samples = len(samples)

            if total_samples == 0:
                return BroadcastAudioChunk(
                    session_id=chunk.session_id,
                    language=chunk.language,
                    source_branch=chunk.source_branch,
                    block_index=chunk.block_index,
                    source_time=chunk.source_time,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    pcm_data=b"",
                )

            if self._is_crossfading and self._fade_old_tail is not None and self.crossfade_samples > 0:
                total_frames = total_samples // self.channels
                remaining_frames = self.crossfade_samples - self._fade_pos
                blend_frames = min(total_frames, remaining_frames)

                if blend_frames > 0:
                    k_start = self._fade_pos
                    k_end = self._fade_pos + blend_frames - 1
                    n = self.crossfade_samples

                    if n > 1:
                        weights = np.linspace(
                            k_start / (n - 1),
                            k_end / (n - 1),
                            blend_frames,
                            dtype=np.float32,
                        )
                    else:
                        weights = np.ones(blend_frames, dtype=np.float32)

                    if self.channels == 1:
                        old_slice = self._fade_old_tail[k_start : k_start + blend_frames].astype(np.float32)
                        new_slice = samples[:blend_frames].astype(np.float32)
                        blended = np.clip(
                            np.round((1.0 - weights) * old_slice + weights * new_slice),
                            -32768,
                            32767,
                        ).astype(np.int16)
                        samples[:blend_frames] = blended
                    else:
                        blend_samples = blend_frames * self.channels
                        old_slice = self._fade_old_tail[
                            k_start * self.channels : (k_start + blend_frames) * self.channels
                        ].reshape(-1, self.channels).astype(np.float32)
                        new_slice = samples[:blend_samples].reshape(-1, self.channels).astype(np.float32)
                        w = weights[:, None]
                        blended = np.clip(
                            np.round((1.0 - w) * old_slice + w * new_slice),
                            -32768,
                            32767,
                        ).astype(np.int16).reshape(-1)
                        samples[:blend_samples] = blended

                    self._fade_pos += blend_frames
                    if self._fade_pos >= self.crossfade_samples:
                        self._is_crossfading = False
                        self._fade_old_tail = None
                        self._fade_pos = 0

            # Store tail of current output stream for future crossfades (do not overwrite if empty)
            if total_samples > 0:
                tail_samples = self.crossfade_samples * self.channels
                if total_samples >= tail_samples:
                    self._last_tail = samples[-tail_samples:].copy()
                else:
                    self._last_tail = samples.copy()

            return BroadcastAudioChunk(
                session_id=chunk.session_id,
                language=chunk.language,
                source_branch=chunk.source_branch,
                block_index=chunk.block_index,
                source_time=chunk.source_time,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
                pcm_data=samples.tobytes(),
            )

    def create_silence_chunk(
        self,
        duration_ms: float = 20.0,
        session_id: str = "",
        language: str = "en",
        block_index: int = 0,
        source_time: float | None = None,
    ) -> BroadcastAudioChunk:
        """Create a silence BroadcastAudioChunk for the active branch."""
        num_frames = int(self.sample_rate * (duration_ms / 1000.0))
        num_samples = num_frames * self.channels
        pcm_data = b"\x00" * (num_samples * 2)
        return BroadcastAudioChunk(
            session_id=session_id,
            language=language,
            source_branch=self.active_branch,
            block_index=block_index,
            source_time=time.monotonic() if source_time is None else source_time,
            sample_rate=self.sample_rate,
            channels=self.channels,
            pcm_data=pcm_data,
        )

    def _trigger_crossfade(self) -> None:
        """Prepare crossfade state using the tail of the previous stream."""
        tail_samples = self.crossfade_samples * self.channels
        if self._last_tail is not None and len(self._last_tail) > 0 and self.crossfade_samples > 0:
            if len(self._last_tail) < tail_samples:
                pad_len = tail_samples - len(self._last_tail)
                self._fade_old_tail = np.pad(self._last_tail, (pad_len, 0), mode="edge")
            else:
                self._fade_old_tail = self._last_tail[-tail_samples:].copy()
            self._fade_pos = 0
            self._is_crossfading = True
        else:
            self._is_crossfading = False
            self._fade_old_tail = None
            self._fade_pos = 0

    def _save_state(self) -> None:
        """Atomically persist current state to disk if state_file_path is configured."""
        if self.state_file_path is not None:
            try:
                self._state.save(self.state_file_path)
            except (OSError, TypeError, ValueError) as e:
                logger.error("Failed to save selector state: %s", e)
