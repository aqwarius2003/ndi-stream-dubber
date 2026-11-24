"""Tests for operator HTTP API and Rich TUI dashboard."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from dubber.config import DubberConfig
from dubber.engine import DubberEngine
from dubber.operator import create_app, generate_dashboard, run_operator_tui
from dubber.types import BranchState


def test_operator_api_status_and_switch(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    engine = DubberEngine(config)
    app = create_app(engine)
    client = TestClient(app)

    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert "active_branch" in data
    assert "alert" in data
    assert data["alert"] is False
    assert "branches" in data
    assert "gemini" in data["branches"]
    assert "soniox" in data["branches"]
    assert "dropped_blocks" in data["branches"]["gemini"]
    assert "dropped_blocks" in data["branches"]["soniox"]
    assert data["branches"]["soniox"]["freshness_ms"] is None

    # Try switching to soniox when not fresh / not ready
    res_switch = client.post("/switch/soniox")
    assert res_switch.status_code == 400
    assert "REJECTED" in res_switch.json()["detail"]

    # Now make soniox ready and fresh
    engine.selector.update_branch_state(
        BranchState("soniox", status="READY", last_audio_time=time.monotonic())
    )
    res_switch_ok = client.post("/switch/soniox")
    assert res_switch_ok.status_code == 200
    assert res_switch_ok.json()["success"] is True
    assert res_switch_ok.json()["active_branch"] == "soniox"


def test_operator_api_stale_and_invalid_switch(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"), freshness_threshold_s=0.5)
    engine = DubberEngine(config)
    app = create_app(engine)
    client = TestClient(app)

    # Invalid branch
    res_invalid = client.post("/switch/invalid_branch")
    assert res_invalid.status_code == 400
    assert "REJECTED" in res_invalid.json()["detail"]

    # Target is READY but stale (last audio 2 seconds ago)
    engine.selector.update_branch_state(
        BranchState("soniox", status="READY", last_audio_time=time.monotonic() - 2.0)
    )
    res_stale = client.post("/switch/soniox")
    assert res_stale.status_code == 400
    assert "REJECTED" in res_stale.json()["detail"]
    assert "NOT FRESH" in res_stale.json()["detail"]

    # Switch to currently active branch (gemini is active by default)
    res_same = client.post("/switch/gemini")
    assert res_same.status_code == 200
    assert res_same.json()["success"] is True
    assert "already active" in res_same.json()["message"]


def test_operator_api_alert_reporting(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    engine = DubberEngine(config)
    app = create_app(engine)
    client = TestClient(app)

    # Put active branch (gemini) into ERROR state
    engine.selector.update_branch_state(
        BranchState("gemini", status="ERROR", error_message="Network timeout")
    )
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["alert"] is True
    assert data["branches"]["gemini"]["status"] == "ERROR"


def test_generate_dashboard(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    engine = DubberEngine(config)
    layout = generate_dashboard(engine)
    assert layout is not None
    assert layout["header"] is not None
    assert layout["main"] is not None
    assert layout["footer"] is not None

    # With alert active
    engine.selector.update_branch_state(BranchState("gemini", status="ERROR"))
    layout_alert = generate_dashboard(engine)
    assert layout_alert is not None


@pytest.mark.asyncio
async def test_run_operator_tui_stop(tmp_path):
    config = DubberConfig(state_file_path=str(tmp_path / "state.json"))
    engine = DubberEngine(config)
    stop_event = asyncio.Event()

    task = asyncio.create_task(run_operator_tui(engine, refresh_rate=10.0, stop_event=stop_event))
    await asyncio.sleep(0.05)
    stop_event.set()
    await task
