"""Abstract Base Class for audio ingest sources."""

from abc import ABC, abstractmethod

from dubber.types import AudioBlock


class AudioSource(ABC):
    """Abstract interface for audio capture sources supplying 16 kHz mono 20 ms blocks."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize and start capturing/reading audio."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop capturing and release resources."""
        ...

    @abstractmethod
    async def read_block(self) -> AudioBlock | None:
        """Read the next 20 ms AudioBlock (320 samples, 16-bit mono PCM).

        Returns None when EOF is reached or the source is stopped.
        """
        ...
