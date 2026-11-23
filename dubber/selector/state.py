"""Persistent state management for DubSelector."""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dubber.types import BranchType

logger = logging.getLogger(__name__)


@dataclass
class SelectorState:
    """State of the dub selector persisted across restarts."""

    active_branch: BranchType = "gemini"
    last_switch_time: float = 0.0
    switch_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any, default_branch: BranchType = "gemini") -> "SelectorState":
        """Reconstruct state from dictionary."""
        if not isinstance(data, dict):
            return cls(active_branch=default_branch)

        active = data.get("active_branch", default_branch)
        if active not in ("gemini", "soniox"):
            active = default_branch

        try:
            last_switch_time = float(data.get("last_switch_time", 0.0))
        except (TypeError, ValueError):
            last_switch_time = 0.0

        try:
            switch_count = int(data.get("switch_count", 0))
        except (TypeError, ValueError):
            switch_count = 0

        raw_meta = data.get("metadata")
        metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}

        return cls(
            active_branch=active,
            last_switch_time=last_switch_time,
            switch_count=switch_count,
            metadata=metadata,
        )

    def save(self, file_path: str | Path) -> None:
        """Atomically save state to file using a temporary file and os.replace."""
        target_path = Path(file_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in the same directory for atomic replace
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix="selector_state_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(temp_path, str(target_path))
        except (OSError, TypeError, ValueError):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    @classmethod
    def load(cls, file_path: str | Path, default_branch: BranchType = "gemini") -> "SelectorState":
        """Load state from file, or return default state if file does not exist or is invalid."""
        path = Path(file_path).resolve()
        if not path.is_file():
            return cls(active_branch=default_branch)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, default_branch=default_branch)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Failed to load selector state from %s (%s). Using default.", path, e)
            return cls(active_branch=default_branch)
