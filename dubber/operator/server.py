"""FastAPI control and monitoring server for PharmaLive Dubber."""

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

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
        "error_message": st.error_message,
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

    @app.get("/health")
    def get_health() -> dict[str, Any]:
        """Health check endpoint returning system status and active branch."""
        alert = engine.selector.is_alert_active()
        return {
            "status": "degraded" if alert else "ok",
            "active_branch": engine.selector.active_branch,
            "alert": alert,
        }

    @app.get("/metrics")
    def get_metrics(format: str | None = None) -> Any:
        """Return operational metrics in JSON or Prometheus text format."""
        now = time.monotonic()
        gemini_info = _get_branch_info(engine, "gemini", now)
        soniox_info = _get_branch_info(engine, "soniox", now)
        alert = engine.selector.is_alert_active()
        active = engine.selector.active_branch

        if format == "prometheus":
            lines = [
                "# HELP dubber_active_branch Active dubbing branch (1 if active, 0 otherwise)",
                "# TYPE dubber_active_branch gauge",
                f'dubber_active_branch{{branch="gemini"}} {1 if active == "gemini" else 0}',
                f'dubber_active_branch{{branch="soniox"}} {1 if active == "soniox" else 0}',
                "# HELP dubber_alert Active branch alert status (1 if in alert, 0 otherwise)",
                "# TYPE dubber_alert gauge",
                f"dubber_alert {1 if alert else 0}",
                "# HELP dubber_latency_seconds Translation latency in seconds",
                "# TYPE dubber_latency_seconds gauge",
                f'dubber_latency_seconds{{branch="gemini"}} {gemini_info["latency_s"]}',
                f'dubber_latency_seconds{{branch="soniox"}} {soniox_info["latency_s"]}',
                "# HELP dubber_freshness_milliseconds Milliseconds since last received audio",
                "# TYPE dubber_freshness_milliseconds gauge",
                f'dubber_freshness_milliseconds{{branch="gemini"}} {gemini_info["freshness_ms"] if gemini_info["freshness_ms"] is not None else -1}',
                f'dubber_freshness_milliseconds{{branch="soniox"}} {soniox_info["freshness_ms"] if soniox_info["freshness_ms"] is not None else -1}',
                "# HELP dubber_dropped_blocks_total Total dropped audio blocks",
                "# TYPE dubber_dropped_blocks_total counter",
                f'dubber_dropped_blocks_total{{branch="gemini"}} {gemini_info["dropped_blocks"]}',
                f'dubber_dropped_blocks_total{{branch="soniox"}} {soniox_info["dropped_blocks"]}',
            ]
            return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

        return {
            "status": "degraded" if alert else "ok",
            "active_branch": active,
            "alert": alert,
            "branches": {
                "gemini": gemini_info,
                "soniox": soniox_info,
            },
            "dropped_blocks": dict(engine.dropped_blocks),
            "total_dropped_blocks": sum(engine.dropped_blocks.values()),
        }

    @app.post("/switch/{branch}")
    def switch_branch(branch: str, force: bool = False) -> dict[str, Any]:
        """Switch broadcast route to target branch upon operator command.

        Rejects switch with HTTP 400 if target branch is stale or not ready.
        """
        target = branch.strip().lower()
        ok, msg = engine.selector.switch_branch(target, force=force)  # type: ignore[arg-type]
        if not ok:
            detail = f"REJECTED: {msg}" if not msg.startswith("REJECTED") else msg
            raise HTTPException(status_code=400, detail=detail)
        return {
            "success": True,
            "message": msg,
            "active_branch": engine.selector.active_branch,
        }

    return app
