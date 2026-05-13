"""
get_deployment_health tool.

Single-call snapshot for "how is this pod doing right now?" — meant for
both the LLM agent (to answer user questions like "is my model drifting?")
and on-call humans poking the agent via chat.

Composes model readiness, baseline, last sanity run, drift history, and
fresh hardware into one dict. Everything it returns is already computed
and cheap to fetch — no heavy work happens here beyond a fresh Jetson
sysfs sample and a Triton ready probe.
"""

import logging
from typing import Any, Dict

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def get_deployment_health() -> Dict[str, Any]:
    """
    Return a one-shot health report for this Helm release.

    Contains: `model` identity + readiness, `baseline` (first-boot
    reference), `current` (most-recent sanity eval + drift score),
    `hardware` (Jetson snapshot), `drift_events` (recent), `alerts`
    (human-readable flags), and `config` (active thresholds).
    """
    try:
        from deployment.health import build_health_report
        report = build_health_report()
        # Surface a short human-readable summary so the agent can stream
        # it verbatim without running additional tools first.
        summary_bits = []
        m = report.get("model") or {}
        if m.get("name"):
            ready = m.get("ready")
            summary_bits.append(f"model {m['name']} {'READY' if ready else 'NOT READY'}")
        baseline = report.get("baseline") or {}
        if baseline.get("inference_p95_ms"):
            summary_bits.append(f"baseline p95 {baseline['inference_p95_ms']:.1f}ms")
        current = report.get("current") or {}
        if current.get("inference_p95_ms"):
            drift = current.get("drift_score")
            if drift is not None:
                summary_bits.append(f"current p95 {current['inference_p95_ms']:.1f}ms (drift {drift:.2f}x)")
            else:
                summary_bits.append(f"current p95 {current['inference_p95_ms']:.1f}ms")
        hw = report.get("hardware") or {}
        if hw.get("junction_temp_c") is not None:
            summary_bits.append(f"Tj {hw['junction_temp_c']:.0f}°C")
        if hw.get("total_power_w") is not None:
            summary_bits.append(f"{hw['total_power_w']:.1f}W")
        if report.get("alerts"):
            summary_bits.append(f"{len(report['alerts'])} alert(s)")

        summary = " | ".join(summary_bits) if summary_bits else "deployment health report"
        return ok(
            summary=summary,
            **report,
        )
    except Exception as e:
        logger.error("get_deployment_health failed: %s", e, exc_info=True)
        return error_response(e, operation="get_deployment_health")


register_tool(
    name="get_deployment_health",
    func=get_deployment_health,
    description=(
        "One-call health snapshot for the current Helm deployment. Returns the "
        "loaded model's readiness, the first-boot latency/thermal baseline, the "
        "most recent scheduled sanity-eval run with drift_score vs baseline, a "
        "fresh Jetson hardware reading (GPU util, junction temp, total power), "
        "recent drift events, and human-readable alerts. Use this tool when the "
        "user asks about overall health, drift, performance over time, or "
        "whether the model is behaving the same as it was at deploy time."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
