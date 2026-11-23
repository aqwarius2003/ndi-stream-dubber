"""Abstract Base Class for audio output sinks."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from dubber.types import BroadcastAudioChunk


class AudioSink(ABC):
    """Abstract interface for audio sinks consuming BroadcastAudioChunk."""

    @abstractmethod
    async def start(self) -> None:
        """Create and initialize output stream / sink."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Flush and safely close the sink."""
        ...

    @abstractmethod
    async def write_chunk(self, chunk: BroadcastAudioChunk) -> None:
        """Write a BroadcastAudioChunk to the sink."""
        ...

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()
