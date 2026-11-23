"""Abstract Base Class for translation workers."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from dubber.types import BranchState, BroadcastAudioChunk


class BaseWorker(ABC):
    """Abstract interface for translation workers (e.g. GeminiWorker, SonioxWorker)."""

    @abstractmethod
    async def start(self) -> None:
        """Start worker background tasks and connect to provider."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop worker background tasks and release resources."""
        ...

    @abstractmethod
    async def get_output_chunk(self) -> BroadcastAudioChunk | None:
        """Retrieve next translated audio chunk for broadcast."""
        ...

    @abstractmethod
    def get_state(self) -> BranchState:
        """Return the current branch state (status, latency, freshness, transcripts)."""
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
