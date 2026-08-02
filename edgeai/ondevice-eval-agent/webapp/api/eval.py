"""Evaluation API routes — datasets, hardware metrics, and result retrieval."""

import logging

from flask import Blueprint, jsonify, request

from utils.errors import (
    BadRequestError,
    NotFoundError,
    create_success_response,
    handle_exceptions,
)

logger = logging.getLogger(__name__)

eval_bp = Blueprint('eval', __name__, url_prefix='/eval')


@eval_bp.route('/datasets', methods=['GET'])
@handle_exceptions("Failed to list datasets")
def list_datasets():
    """
    List available evaluation datasets.

    Response:
        {
            "success": true,
            "datasets": [
                {"name": "general_knowledge", "item_count": 60, "categories": [...]}
            ]
        }
    """
    from eval.dataset_loader import list_datasets as _list_datasets

    datasets = _list_datasets()
    return jsonify(create_success_response({
        "datasets": datasets,
        "count": len(datasets),
    }))


@eval_bp.route('/datasets/<name>', methods=['GET'])
@handle_exceptions("Failed to load dataset")
def get_dataset(name: str):
    """
    Preview a dataset (first 20 items).

    Response:
        {
            "success": true,
            "name": "general_knowledge",
            "items": [...],
            "total_items": 60,
            "preview": true
        }
    """
    from eval.dataset_loader import load_dataset

    try:
        items = load_dataset(name)
    except ValueError as e:
        raise NotFoundError(str(e))

    preview_count = 20
    return jsonify(create_success_response({
        "name": name,
        "items": items[:preview_count],
        "total_items": len(items),
        "preview": len(items) > preview_count,
    }))


@eval_bp.route('/hardware', methods=['GET'])
@handle_exceptions("Failed to read hardware metrics")
def get_hardware_metrics():
    """
    Get a single Jetson hardware metrics snapshot.

    Response:
        {
            "success": true,
            "snapshot": {
                "gpu_util_pct": 45.2,
                "cpu_temp_c": 42.1,
                "junction_temp_c": 45.5,
                "vdd_gpu_soc_w": 3.2,
                "total_power_w": 8.1,
                ...
            }
        }
    """
    from eval.hardware_metrics import read_snapshot

    snapshot = read_snapshot()
    return jsonify(create_success_response({
        "snapshot": snapshot.to_dict(),
    }))


@eval_bp.route('/results', methods=['GET'])
@handle_exceptions("Failed to list results")
def list_eval_results():
    """
    List saved evaluation/benchmark results for a session.

    Query params:
        session_id: Required session identifier
        type: Optional filter (benchmark, eval, comparison)

    Response:
        {
            "success": true,
            "results": [...],
            "count": 3
        }
    """
    session_id = request.args.get("session_id")
    if not session_id:
        raise BadRequestError("session_id query parameter is required")

    result_type = request.args.get("type")

    from eval.result_store import list_results

    results = list_results(session_id, result_type=result_type)
    return jsonify(create_success_response({
        "results": results,
        "count": len(results),
    }))


@eval_bp.route('/results/<filename>', methods=['GET'])
@handle_exceptions("Failed to load result")
def get_eval_result(filename: str):
    """
    Load a specific saved result.

    Query params:
        session_id: Required session identifier

    Response:
        {
            "success": true,
            "result": {...}
        }
    """
    session_id = request.args.get("session_id")
    if not session_id:
        raise BadRequestError("session_id query parameter is required")

    from eval.result_store import load_result

    try:
        result = load_result(session_id, filename)
    except FileNotFoundError as e:
        raise NotFoundError(str(e))

    return jsonify(create_success_response({
        "result": result,
    }))
