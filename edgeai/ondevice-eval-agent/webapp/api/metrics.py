"""
Prometheus `/metrics` endpoint.

Central Prometheus scrapes every Helm release on this path and joins
the per-deployment signals via the `ondevice_eval_deployment_info`
gauge's labels (model_name, mlflow_run_id, deployment_id).

Returns 404 when `prometheus_client` isn't installed or deployment
features are off — that way a scrape misconfiguration degrades
gracefully instead of 500-ing.
"""

from __future__ import annotations

import logging

from flask import Blueprint, Response, jsonify

logger = logging.getLogger(__name__)

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics", methods=["GET"])
def metrics():
    try:
        from deployment import metrics as prom
    except Exception as e:
        logger.warning("metrics endpoint: deployment.metrics import failed: %s", e)
        return Response("metrics unavailable", status=404)

    if not prom.available():
        return Response("metrics unavailable", status=404)

    try:
        body = prom.render()
    except Exception as e:
        logger.exception("metrics render failed: %s", e)
        return jsonify({"error": str(e)}), 500

    return Response(body, mimetype=prom.content_type())
