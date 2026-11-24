"""Report generation for latency measurements against the 3.0s SLA target."""

import numpy as np

from dubber.measure.latency import LatencyResult


def generate_latency_report(
    results: list[LatencyResult],
    threshold_s: float = 3.0,
    execution_type: str = "Live",
    provider_mode: str = "live",
) -> str:
    """Generate a clean, human-readable latency evaluation report.

    Calculates p50, p95, min, max, and pass/fail against 3.0s threshold.
    Identifies file runs vs live broadcasts, and mock simulations vs real APIs.
    """
    lines = [
        "=" * 68,
        "               PHARMALIVE DUBBER LATENCY REPORT",
        "=" * 68,
        f"Execution Type:        {execution_type}",
        f"Provider Mode:         {provider_mode}",
    ]

    if not results:
        lines.append("No latency measurements available.")
        lines.append("=" * 68)
        return "\n".join(lines)

    latencies = [r.latency_s for r in results]
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100.0

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    max_lat = float(np.max(latencies))
    min_lat = float(np.min(latencies))

    status_str = "PASS" if passed_count == total_count else "FAIL"

    lines.append(f"Overall Status:        {status_str}")
    lines.append(f"Total Evaluated:       {total_count}")
    lines.append(f"Passed SLA:            {passed_count}/{total_count} ({pass_rate:.1f}%)")
    lines.append(f"Latency SLA Target:    <= {threshold_s:.2f}s")
    lines.append("-" * 68)
    lines.append(f"Median latency (p50):  {p50:.3f}s")
    lines.append(f"95th percentile (p95): {p95:.3f}s")
    lines.append(f"Min latency:           {min_lat:.3f}s")
    lines.append(f"Max latency:           {max_lat:.3f}s")
    lines.append("-" * 68)
    lines.append(
        f"{'Phrase ID':<20} {'t_in (s)':<10} {'t_out (s)':<10} {'Latency (s)':<14} {'Status'}"
    )
    lines.append("-" * 68)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.phrase_id:<20} {r.t_in:<10.3f} {r.t_out:<10.3f} {r.latency_s:<14.3f} {flag}"
        )
    lines.append("=" * 68)
    return "\n".join(lines)
