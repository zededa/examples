"""
Inference operations for Model Server Client via gRPC.

This module handles sending inference requests and processing responses
from both Triton and OpenVINO inference servers using the KServe v2
gRPC protocol.  Tensor data is transferred in binary form, avoiding
the JSON serialization overhead of the REST API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray
from tritonclient.utils import InferenceServerException

from .config import DEFAULT_INFERENCE_TIMEOUT_SECONDS
from .exceptions import InferenceError

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_INPUT_NAME: Final[str] = "input"
_DEFAULT_DATATYPE: Final[str] = "FP32"
_DEFAULT_TOP_K: Final[int] = 5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InferenceRequest:
    """
    Structured inference request for KServe v2 gRPC API.

    Encapsulates all data needed for an inference request.
    """
    model_name: str
    input_name: str
    input_shape: List[int]
    input_data: NDArray[np.floating[Any]]
    datatype: str = _DEFAULT_DATATYPE

    def to_grpc_inputs(self) -> List[grpcclient.InferInput]:
        """Build gRPC InferInput objects from this request."""
        infer_input = grpcclient.InferInput(
            self.input_name,
            self.input_shape,
            self.datatype,
        )
        infer_input.set_data_from_numpy(self.input_data.astype(np.float32))
        return [infer_input]


@dataclass
class InferenceResult:
    """Structured inference result."""
    model_name: str
    outputs: List[Dict[str, Any]]
    latency: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_name": self.model_name,
            "outputs": self.outputs,
        }
        if self.latency is not None:
            result["latency"] = self.latency
        return result


@dataclass
class ClassificationResult:
    """Classification prediction result."""
    model_name: str
    timestamp: str
    num_classes: int
    output_name: str
    output_shape: List[int]
    predictions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "top_predictions": self.predictions,
        }


# =============================================================================
# Inference Runner
# =============================================================================

class InferenceRunner:
    """
    Handles inference request execution and response processing via gRPC.

    Uses tritonclient.grpc to send numpy arrays directly over gRPC,
    eliminating JSON serialization overhead for tensor data.
    """

    __slots__ = ("_grpc_client", "_timeout", "_class_names")

    def __init__(
        self,
        grpc_client: grpcclient.InferenceServerClient,
        timeout: int = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the inference runner.

        Args:
            grpc_client: gRPC inference-server client instance.
            timeout: Inference request timeout in seconds.
        """
        self._grpc_client = grpc_client
        self._timeout = timeout
        self._class_names: Optional[List[str]] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def class_names(self) -> Optional[List[str]]:
        """Get class names for labeling predictions."""
        return self._class_names

    @class_names.setter
    def class_names(self, value: Optional[List[str]]) -> None:
        """Set class names for labeling predictions."""
        self._class_names = value

    # =========================================================================
    # Public API - Inference
    # =========================================================================

    def send_inference_request(
        self,
        image_array: NDArray[np.floating[Any]],
        model_name: str,
        input_spec: Dict[str, Any],
        server_type: str,
        measure_latency: bool = False,
    ) -> Dict[str, Any]:
        """
        Send inference request to inference server via gRPC.

        Args:
            image_array: Preprocessed image array with batch dimension.
            model_name: Name of the model.
            input_spec: Model input specification.
            server_type: Server type ('triton', 'openvino', 'unknown').
            measure_latency: Whether to include request latency in result.

        Returns:
            Raw inference response dict.

        Raises:
            InferenceError: If inference fails.
        """
        request = InferenceRequest(
            model_name=model_name,
            input_name=input_spec.get("name", _DEFAULT_INPUT_NAME),
            input_shape=list(image_array.shape),
            input_data=image_array,
            datatype=input_spec.get("datatype", _DEFAULT_DATATYPE),
        )

        result = self._send_grpc_inference(request, measure_latency)
        if result is not None:
            return result

        raise InferenceError(
            f"gRPC inference failed for model {model_name}",
            model_name=model_name,
        )

    def process_prediction(
        self,
        response: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process the prediction response from inference server.

        For classification models, applies softmax and returns top-k predictions.
        For non-classification outputs, returns raw output info.

        Raises:
            InferenceError: If response format is invalid.
        """
        if not response or "outputs" not in response:
            raise InferenceError(
                f"Invalid response format for model {model_name}",
                model_name=model_name,
            )

        try:
            output_data = self._extract_output_data(response)
            scores = self._reshape_scores(output_data)

            if self._is_classification_output(scores):
                return self._process_classification(
                    scores,
                    output_data["name"],
                    output_data["shape"],
                    model_name,
                )

            return self._create_raw_output_result(
                scores, output_data["name"], output_data["shape"], model_name
            )

        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise InferenceError(
                f"Error processing prediction: {e}",
                model_name=model_name,
                details={"cause": str(e)},
            ) from e

    # =========================================================================
    # Private - gRPC Inference
    # =========================================================================

    def _send_grpc_inference(
        self,
        request: InferenceRequest,
        measure_latency: bool,
    ) -> Optional[Dict[str, Any]]:
        """Send inference using gRPC with binary tensor transfer."""
        try:
            grpc_inputs = request.to_grpc_inputs()

            # Request all outputs from the model
            # (passing None for outputs requests all available outputs)
            start_time = time.perf_counter()
            grpc_result = self._grpc_client.infer(
                model_name=request.model_name,
                inputs=grpc_inputs,
                client_timeout=self._timeout,
            )
            latency = time.perf_counter() - start_time

            # Convert gRPC result to the dict format expected downstream
            result = self._grpc_result_to_dict(grpc_result, request.model_name)

            if measure_latency:
                result["latency"] = latency

            logger.debug(
                f"gRPC inference successful for {request.model_name} "
                f"({latency*1000:.1f}ms)"
            )
            return result

        except InferenceServerException as e:
            logger.warning(f"gRPC inference failed for {request.model_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"gRPC inference error for {request.model_name}: {e}")
            return None

    def _grpc_result_to_dict(
        self,
        grpc_result: grpcclient.InferResult,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Convert a gRPC InferResult into the dict format matching the
        KServe v2 REST inference response used by downstream code.
        """
        outputs: List[Dict[str, Any]] = []

        # Get the result's response object to enumerate output names
        response = grpc_result.get_response()
        if hasattr(response, "outputs"):
            for out_meta in response.outputs:
                out_name = out_meta.name
                out_data = grpc_result.as_numpy(out_name)
                outputs.append({
                    "name": out_name,
                    "shape": list(out_data.shape),
                    "datatype": out_meta.datatype,
                    "data": out_data.flatten().tolist(),
                })
        else:
            # Fallback: try output_0
            try:
                out_data = grpc_result.as_numpy("output_0")
                outputs.append({
                    "name": "output_0",
                    "shape": list(out_data.shape),
                    "datatype": "FP32",
                    "data": out_data.flatten().tolist(),
                })
            except Exception:
                pass

        return {
            "model_name": model_name,
            "outputs": outputs,
        }

    # =========================================================================
    # Private - Response Processing
    # =========================================================================

    def _extract_output_data(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract first output data from response."""
        outputs = response["outputs"]

        if not isinstance(outputs, list) or len(outputs) == 0:
            raise ValueError(f"Unexpected outputs format: {type(outputs)}")

        output = outputs[0]
        return {
            "name": output.get("name", "output"),
            "shape": output.get("shape", []),
            "data": output.get("data", []),
        }

    def _reshape_scores(self, output_data: Dict[str, Any]) -> NDArray:
        """Reshape prediction scores based on output shape."""
        scores = np.array(output_data["data"])
        shape = output_data["shape"]

        if shape:
            scores = scores.reshape(shape)

        if len(scores.shape) == 2 and scores.shape[0] == 1:
            scores = scores[0]

        return scores

    @staticmethod
    def _is_classification_output(scores: NDArray) -> bool:
        """Check if output looks like classification (1D array with multiple values)."""
        return len(scores.shape) == 1 and len(scores) > 1

    def _create_raw_output_result(
        self,
        scores: NDArray,
        output_name: str,
        output_shape: List[int],
        model_name: Optional[str],
    ) -> Dict[str, Any]:
        """Create result dict for non-classification outputs."""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "output_name": output_name,
            "output_shape": output_shape,
            "raw_output": scores.tolist() if hasattr(scores, "tolist") else scores,
            "top_predictions": [],
        }

    # =========================================================================
    # Private - Classification Processing
    # =========================================================================

    def _process_classification(
        self,
        scores: NDArray,
        output_name: str,
        output_shape: List[int],
        model_name: Optional[str],
    ) -> Dict[str, Any]:
        """Process classification model output."""
        probabilities = self._softmax(scores)

        num_classes = len(probabilities)
        top_k = min(_DEFAULT_TOP_K, num_classes)
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]

        predictions = [
            self._create_prediction_entry(i, int(idx), float(prob))
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs))
        ]

        return ClassificationResult(
            model_name=model_name or "unknown",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            num_classes=num_classes,
            output_name=output_name,
            output_shape=output_shape,
            predictions=predictions,
        ).to_dict()

    @staticmethod
    def _softmax(scores: NDArray) -> NDArray:
        """Apply softmax normalization with numerical stability."""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def _create_prediction_entry(
        self,
        rank: int,
        class_id: int,
        probability: float,
    ) -> Dict[str, Any]:
        """Create a single prediction entry with optional class name."""
        class_name = (
            self._class_names[class_id]
            if self._class_names and 0 <= class_id < len(self._class_names)
            else f"Class_{class_id}"
        )

        return {
            "rank": rank + 1,
            "class_id": class_id,
            "confidence": probability,
            "probability": probability,
            "class_name": class_name,
        }


__all__ = [
    "InferenceRunner",
    "InferenceRequest",
    "InferenceResult",
    "ClassificationResult",
]
