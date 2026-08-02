"""
Per-Helm-release deployment awareness.

This package turns the business-logic image into a deployment-aware
eval sidecar: it introspects the single loaded model at boot, captures
a golden latency/thermal baseline, runs scheduled sanity evals, and
exposes fleet-observable signals via Prometheus `/metrics`.

Entry point: `bootstrap.start()` is called once from `app.py` after
blueprints register. Everything else runs in daemon threads so startup
is never blocked.
"""

from .bootstrap import start as start_bootstrap
from .store import DeploymentStore, get_store
from .health import build_health_report

__all__ = [
    "start_bootstrap",
    "DeploymentStore",
    "get_store",
    "build_health_report",
]
