"""Rich TUI dashboard and interactive console interface for PharmaLive Dubber."""

import asyncio
import logging
import sys
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from dubber.engine import DubberEngine

logger = logging.getLogger(__name__)


def _poll_key() -> str | None:
    """Non-blocking check for keyboard input on Windows console."""
    if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
        return None
    try:
        import msvcrt

        if msvcrt.kbhit():
            ch = msvcrt.getch()
            return ch.decode("utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return None
    return None


def generate_dashboard(engine: DubberEngine) -> Layout:
    """Generate a Rich Layout representing the live operator dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    lang = getattr(getattr(engine, "config", None), "language", "en").upper()
    active = engine.selector.active_branch.upper()
    alert = (
        " [bold red]ALERT: ACTIVE BRANCH FAILED[/bold red]"
        if engine.selector.is_alert_active()
        else ""
    )

    header_panel = Panel(
        f"[bold white]PharmaLive Dubber Control[/bold white] | "
        f"Output: [bold cyan]Dub_{lang}[/bold cyan] -> [bold yellow]{active}[/bold yellow]{alert}",
        style="blue",
    )
    layout["header"].update(header_panel)

    table = Table(title="Translation Branches Status", expand=True)
    table.add_column("Branch", style="bold")
    table.add_column("Status")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Freshness", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Last Transcript")

    now = time.monotonic()
    for branch in ("gemini", "soniox"):
        st = engine.selector.get_branch_state(branch)  # type: ignore[arg-type]
        worker = getattr(engine, f"{branch}_worker", None)
        if worker is not None and hasattr(worker, "get_state"):
            w_st = worker.get_state()
            if w_st.last_audio_time > st.last_audio_time or (
                w_st.last_transcript and not st.last_transcript
            ):
                st = w_st

        age_ms = int((now - st.last_audio_time) * 1000) if st.last_audio_time > 0 else 9999
        freshness_str = f"{age_ms} ms" if st.last_audio_time > 0 else "N/A"
        dropped_count = max(engine.dropped_blocks.get(branch, 0), st.dropped_blocks_count)

        if st.status in ("READY", "RUNNING"):
            status_color = "green"
        elif st.status == "STARTING":
            status_color = "yellow"
        else:
            status_color = "red"

        table.add_row(
            st.branch.capitalize(),
            f"[{status_color}]{st.status}[/{status_color}]",
            f"{st.latency_s:.2f}s",
            freshness_str,
            str(dropped_count),
            st.last_transcript[:60] if st.last_transcript else "-",
        )

    layout["main"].update(table)
    footer_panel = Panel(
        "Commands: [1/g] Switch to Gemini | [2/s] Switch to Soniox | [q] Quit",
        style="white",
    )
    layout["footer"].update(footer_panel)
    return layout


async def run_operator_tui(
    engine: DubberEngine,
    refresh_rate: float = 4.0,
    stop_event: asyncio.Event | None = None,
    console: Console | None = None,
) -> None:
    """Run interactive operator console dashboard loop."""
    if console is None:
        console = Console()

    delay = 1.0 / max(1.0, refresh_rate)
    with Live(generate_dashboard(engine), refresh_per_second=refresh_rate, console=console) as live:
        try:
            while stop_event is None or not stop_event.is_set():
                key = _poll_key()
                if key:
                    k = key.lower()
                    if k in ("1", "g"):
                        engine.selector.switch_branch("gemini")
                    elif k in ("2", "s"):
                        engine.selector.switch_branch("soniox")
                    elif k == "q":
                        if stop_event is not None:
                            stop_event.set()
                        break

                live.update(generate_dashboard(engine))
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass
