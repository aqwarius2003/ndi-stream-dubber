from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from dubber import cli
from dubber.config import DubberConfig


def test_web_session_uses_selected_ndi_route_and_one_live_provider():
    from dubber.operator.web import create_web_app

    with (
        patch("dubber.operator.web.discover_sources", return_value=["PC (vMix - Output 1)"]),
        patch("dubber.engine.DubberEngine.start", new_callable=AsyncMock) as start,
        patch("dubber.engine.DubberEngine.stop", new_callable=AsyncMock) as stop,
        TestClient(create_web_app(DubberConfig(gemini_api_key="private-test-key"))) as client,
    ):
        assert cli.build_parser().parse_args(["web"]).command == "web"
        assert client.get("/").status_code == 200
        assert client.get("/api/sources").json() == ["PC (vMix - Output 1)"]
        args = {"source": "PC (vMix - Output 1)", "direction": "en-ru"}
        assert client.post("/api/start", json=args).status_code == 200
        engine = client.app.state.engine
        assert engine.source.source_name == args["source"]
        assert engine.sink.sink_name == "Dub_RU"
        assert engine.gemini_worker.target_language == "ru"
        assert engine.gemini_worker.mode == "live"
        assert engine.soniox_worker is None
        assert client.post("/api/start", json=args).status_code == 409
        assert "private-test-key" not in client.get("/api/status").text
        assert client.get("/api/status").json()["direction"] == "en-ru"
        assert client.post("/api/stop").status_code == 200
        assert not client.get("/api/status").json()["running"]
        start.assert_awaited_once()
        stop.assert_awaited_once()


@pytest.mark.parametrize("key,source,code", [(None, "PC (vMix)", 400),
                                                ("key", "PC (Dub_EN)", 400)])
def test_invalid_live_session_never_starts(key, source, code):
    from dubber.operator.web import create_web_app

    with TestClient(create_web_app(DubberConfig(gemini_api_key=key))) as client:
        assert client.post("/api/start", json={"source": source, "direction": "ru-en"}).status_code == code
        assert not client.get("/api/status").json()["running"]
