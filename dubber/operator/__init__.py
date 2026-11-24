"""Operator interfaces for PharmaLive Dubber: Rich TUI & HTTP server."""

from dubber.operator.cli import generate_dashboard, run_operator_tui
from dubber.operator.server import create_app

__all__ = [
    "create_app",
    "generate_dashboard",
    "run_operator_tui",
]
