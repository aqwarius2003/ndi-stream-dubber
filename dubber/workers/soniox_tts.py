"""Soniox Streaming Real-Time Text-to-Speech (TTS) client."""

import base64
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class SonioxTTSError(RuntimeError):
    """Exception raised when Soniox TTS encounters an API or synthesis error."""

    def __init__(
        self,
        error_type: str = "unknown_error",
        error_code: int | None = None,
        error_message: str = "Unknown Soniox TTS error",
    ) -> None:
        super().__init__(f"{error_type} ({error_code}): {error_message}")
        self.error_type = error_type
        self.error_code = error_code
        self.error_message = error_message


class SonioxStreamingTTS:
    """Client for Soniox real-time WebSocket TTS streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_url: str = "wss://tts-rt.soniox.com/tts-websocket",
        model: str = "tts-rt-v2",
        voice: str = "Adrian",
        sample_rate: int = 48000,
        language: str = "en",
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SONIOX_API_KEY")
        self.endpoint_url = endpoint_url
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.language = language
        self.connect_timeout_s = connect_timeout_s

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream raw PCM 48kHz audio chunks for given text incrementally.

        Yields PCM audio chunks as they arrive from WebSocket messages,
        including any audio payload delivered alongside audio_end=True.
        """
        if not text or not text.strip():
            return

        stream_id = uuid.uuid4().hex
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        connect_kwargs: dict[str, Any] = {}
        if headers:
            connect_kwargs["additional_headers"] = headers

        try:
            async with websockets.connect(
                self.endpoint_url,
                open_timeout=self.connect_timeout_s,
                **connect_kwargs,
            ) as ws:
                # 1. Send configuration frame
                config_frame = {
                    "stream_id": stream_id,
                    "model": self.model,
                    "language": self.language,
                    "voice": self.voice,
                    "audio_format": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                }
                await ws.send(json.dumps(config_frame))

                # 2. Send text streaming payload with text_end=True
                text_frame = {
                    "stream_id": stream_id,
                    "text": text,
                    "text_end": True,
                }
                await ws.send(json.dumps(text_frame))

                # 3. Receive audio frames incrementally
                async for raw_msg in ws:
                    try:
                        if isinstance(raw_msg, bytes):
                            data = json.loads(raw_msg.decode("utf-8"))
                        else:
                            data = json.loads(raw_msg)
                    except Exception as parse_exc:  # noqa: BLE001
                        logger.debug("Failed parsing Soniox TTS message: %s", parse_exc)
                        continue

                    if "error_type" in data or "error_message" in data:
                        raise SonioxTTSError(
                            error_type=data.get("error_type", "unknown_error"),
                            error_code=data.get("error_code"),
                            error_message=data.get(
                                "error_message", "Unknown Soniox TTS error"
                            ),
                        )

                    if data.get("stream_id") != stream_id:
                        continue

                    audio_b64 = data.get("audio")
                    if audio_b64:
                        pcm_chunk = base64.b64decode(audio_b64)
                        if pcm_chunk:
                            yield pcm_chunk

                    if data.get("audio_end", False) is True:
                        break

        except SonioxTTSError:
            raise
        except Exception as exc:
            logger.error("SonioxStreamingTTS communication failed: %s", exc)
            raise RuntimeError(f"Soniox TTS failed: {exc}") from exc

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text and return full combined PCM bytes."""
        chunks: list[bytes] = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
