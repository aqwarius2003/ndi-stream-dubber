"""Unit tests for SonioxStreamingTTS client."""

import asyncio
import base64
import json

import pytest
import websockets

from dubber.workers.soniox_tts import SonioxStreamingTTS, SonioxTTSError


@pytest.mark.asyncio
async def test_soniox_streaming_tts_success():
    """Verify SonioxStreamingTTS sends config/text and yields streamed audio chunks."""
    received_headers = {}
    received_messages = []

    async def mock_tts_server(websocket):
        nonlocal received_headers
        # websockets headers
        if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
            received_headers = dict(websocket.request.headers)
        elif hasattr(websocket, "request_headers"):
            received_headers = dict(websocket.request_headers)

        # Receive config
        config_msg = await websocket.recv()
        received_messages.append(json.loads(config_msg))

        # Receive text
        text_msg = await websocket.recv()
        received_messages.append(json.loads(text_msg))

        stream_id = received_messages[0]["stream_id"]

        # Stream chunk 1
        await websocket.send(
            json.dumps({
                "stream_id": stream_id,
                "audio": base64.b64encode(b"pcm_data_first_chunk").decode("ascii"),
                "audio_end": False,
            })
        )
        await asyncio.sleep(0.01)

        # Stream chunk 2 with audio_end=True
        await websocket.send(
            json.dumps({
                "stream_id": stream_id,
                "audio": base64.b64encode(b"pcm_data_final_chunk").decode("ascii"),
                "audio_end": True,
            })
        )

    server = await websockets.serve(mock_tts_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}"

    try:
        tts = SonioxStreamingTTS(
            api_key="test_key_123",
            endpoint_url=url,
            model="tts-rt-v2",
            voice="Adrian",
        )

        chunks = []
        async for chunk in tts.synthesize_stream("Hello world"):
            chunks.append(chunk)

        assert chunks == [b"pcm_data_first_chunk", b"pcm_data_final_chunk"]

        # Verify config message
        cfg = received_messages[0]
        assert cfg["model"] == "tts-rt-v2"
        assert cfg["language"] == "en"
        assert cfg["voice"] == "Adrian"
        assert cfg["audio_format"] == "pcm_s16le"
        assert cfg["sample_rate"] == 48000
        assert "stream_id" in cfg

        # Verify text message
        txt = received_messages[1]
        assert txt["stream_id"] == cfg["stream_id"]
        assert txt["text"] == "Hello world"
        assert txt["text_end"] is True

        # Verify Authorization header
        auth_header = received_headers.get("authorization", "") or received_headers.get("Authorization", "")
        assert "Bearer test_key_123" in auth_header

    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_soniox_streaming_tts_server_error():
    """Verify SonioxStreamingTTS raises SonioxTTSError on server error payload."""
    async def mock_error_server(websocket):
        await websocket.recv()  # config
        await websocket.recv()  # text
        await websocket.send(
            json.dumps({
                "error_type": "quota_exceeded",
                "error_code": 402,
                "error_message": "Account out of credits",
            })
        )

    server = await websockets.serve(mock_error_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}"

    try:
        tts = SonioxStreamingTTS(
            api_key="test_key_123",
            endpoint_url=url,
        )

        with pytest.raises(SonioxTTSError) as exc_info:
            async for _ in tts.synthesize_stream("Fail me"):
                pass

        assert "Account out of credits" in str(exc_info.value)
        assert exc_info.value.error_type == "quota_exceeded"
    finally:
        server.close()
        await server.wait_closed()
