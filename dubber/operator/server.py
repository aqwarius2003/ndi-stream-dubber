"""FastAPI control and monitoring server for PharmaLive Dubber."""

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException

from dubber.engine import DubberEngine
from dubber.types import BranchType

logger = logging.getLogger(__name__)


def _get_branch_info(engine: DubberEngine, branch: BranchType, now: float) -> dict[str, Any]:
    """Retrieve combined state and metrics for a translation branch."""
    st = engine.selector.get_branch_state(branch)
    worker = getattr(engine, f"{branch}_worker", None)
    if worker is not None and hasattr(worker, "get_state"):
        w_st = worker.get_state()
        if w_st.last_audio_time > st.last_audio_time or (
            w_st.last_transcript and not st.last_transcript
        ):
            st = w_st

    freshness_ms = (
        max(0, round((now - st.last_audio_time) * 1000))
        if st.last_audio_time > 0
        else None
    )
    dropped = max(
        engine.dropped_blocks.get(branch, 0),
        st.dropped_blocks_count,
    )
    return {
        "status": st.status,
        "latency_s": round(st.latency_s, 3),
        "freshness_ms": freshness_ms,
        "last_transcript": st.last_transcript,
        "dropped_blocks": dropped,
    }


def create_app(engine: DubberEngine) -> FastAPI:
    """Create and configure FastAPI operator application."""
    app = FastAPI(title="PharmaLive Dubber Operator API")

    @app.get("/status")
    def get_status() -> dict[str, Any]:
        """Return current router state, failure alerts, and branch metrics."""
        now = time.monotonic()
        return {
            "active_branch": engine.selector.active_branch,
            "alert": engine.selector.is_alert_active(),
            "branches": {
                "gemini": _get_branch_info(engine, "gemini", now),
                "soniox": _get_branch_info(engine, "soniox", now),
            },
        }

    @app.post("/switch/{branch}")
    def switch_branch(branch: str, force: bool = False) -> dict[str, Any]:
        """Switch broadcast route to target branch upon operator command.

        Rejects switch with HTTP 400 if target branch is stale or not ready.
        """
        ok, msg = engine.selector.switch_branch(branch, force=force)  # type: ignore[arg-type]
        if not ok:
            detail = f"REJECTED: {msg}" if not msg.startswith("REJECTED") else msg
            raise HTTPException(status_code=400, detail=detail)
        return {
            "success": True,
            "message": msg,
            "active_branch": engine.selector.active_branch,
        }

    return app
