"""
Tool catalog.

Each tool is in its own file for easy maintenance and extension.
Import from here to pull in all tool functions; registration happens
as a side effect of `tools.registry` importing this package.
"""

from .list_models import list_available_models
from .model_metadata import get_model_metadata
from .model_config import get_model_config
from .model_inputs import get_model_input_requirements
from .model_outputs import get_model_output_interpretation
from .model_type import analyze_model_type
from .server_status import get_server_status
from .api_examples import get_api_examples
from .integration_guide import get_frontend_integration_guide
from .recommendations import recommend_next_steps
from .run_inference import run_inference, list_processing_types
from .inference_latency import get_inference_latency
from .web_search import web_search, search_model_info
from .view_image import view_image, analyze_inference_result
from .check_model_ready import check_model_ready
from .all_model_outputs import get_all_model_outputs
from .clear_model_cache import clear_model_cache
from .configure_preprocessing import configure_preprocessing
from .compare_models import compare_models
from .detr_inference import run_detr_inference
from .batch_model_status import batch_model_status
from .manage_class_names import manage_class_names
from .llm_list_models import llm_list_models
from .llm_performance import llm_get_performance
from .llm_inference import llm_inference
from .probe_model_io import probe_model_io
from .diagnose_failed_models import diagnose_failed_models
from .fix_model_config import fix_model_config
from .llm_run_benchmark import llm_run_benchmark
from .llm_evaluate import llm_evaluate
from .llm_compare_models import llm_compare_models
from .deployment_health import get_deployment_health

__all__ = [
    "list_available_models",
    "get_model_metadata",
    "get_model_config",
    "get_model_input_requirements",
    "get_model_output_interpretation",
    "analyze_model_type",
    "get_server_status",
    "get_api_examples",
    "get_frontend_integration_guide",
    "recommend_next_steps",
    "run_inference",
    "list_processing_types",
    "get_inference_latency",
    "web_search",
    "search_model_info",
    "view_image",
    "analyze_inference_result",
    "check_model_ready",
    "get_all_model_outputs",
    "clear_model_cache",
    "configure_preprocessing",
    "compare_models",
    "run_detr_inference",
    "batch_model_status",
    "manage_class_names",
    "llm_list_models",
    "llm_get_performance",
    "llm_inference",
    "probe_model_io",
    "diagnose_failed_models",
    "fix_model_config",
    "llm_run_benchmark",
    "llm_evaluate",
    "llm_compare_models",
    "get_deployment_health",
]
