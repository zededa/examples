"""Tensor formatting and summary utilities."""

import base64
from typing import Any, Dict, List, Optional

import numpy as np


def format_tensor_shape(shape: List[int]) -> str:
    """Format tensor shape for display."""
    return f"[{', '.join(str(dim) if dim > 0 else '?' for dim in shape)}]"


def get_tensor_summary(
    array: np.ndarray,
    max_values: int = 10,
    include_full_values: bool = True
) -> Dict[str, Any]:
    """Get a summary of tensor data for display.
    
    Args:
        array: numpy array to summarize
        max_values: number of values for preview (deprecated, kept for compatibility)
        include_full_values: if True, include complete tensor values
        
    Performance Note:
        Uses base64-encoded binary format for large tensors instead of tolist()
        to avoid creating millions of Python float objects (10-100x faster).
    """
    flat = array.flatten()
    total = len(flat)
    
    # Handle empty arrays
    if total == 0:
        return {
            'values_preview': [],
            'total_elements': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'dtype': str(array.dtype),
            'shape': list(array.shape)
        }
    
    # Preview for backward compatibility (small number of values is fine)
    if total <= max_values:
        values = [f"{v:.4f}" for v in flat]
    else:
        first_values = [f"{v:.4f}" for v in flat[:max_values//2]]
        last_values = [f"{v:.4f}" for v in flat[-(max_values//2):]]
        values = first_values + ['...'] + last_values
    
    result: Dict[str, Any] = {
        'values_preview': values,
        'total_elements': total,
        'min': float(flat.min()),
        'max': float(flat.max()),
        'mean': float(flat.mean()),
        'std': float(flat.std()),
        'dtype': str(array.dtype),
        'shape': list(array.shape)
    }
    
    # Include full tensor values for building applications
    if include_full_values:
        # Limit full values to prevent OOM with large tensors (e.g., segmentation masks)
        MAX_TENSOR_ELEMENTS = 50000  # ~200KB when serialized
        total_elements = array.size
        
        if total_elements <= MAX_TENSOR_ELEMENTS:
            # For small tensors, use efficient base64 encoding instead of tolist()
            # This avoids creating Python float objects and is 10-100x faster
            arr_float32 = array.astype(np.float32)
            result['full_values_base64'] = base64.b64encode(arr_float32.tobytes()).decode('ascii')
            result['full_values_encoding'] = 'base64_float32_littleendian'
            result['full_values_shape'] = list(array.shape)
            result['full_values_truncated'] = False
        else:
            # For large tensors, provide a flattened sample and metadata
            result['full_values_truncated'] = True
            result['total_elements'] = int(total_elements)
            result['truncation_reason'] = f"Tensor too large ({total_elements:,} elements). Showing first {MAX_TENSOR_ELEMENTS:,}."
            # Flatten and take first N elements, use base64 encoding
            flat_sample = flat[:MAX_TENSOR_ELEMENTS].astype(np.float32)
            result['full_values_sample_base64'] = base64.b64encode(flat_sample.tobytes()).decode('ascii')
            result['full_values_encoding'] = 'base64_float32_littleendian'
            result['sample_shape'] = f"Flattened first {len(flat_sample)} of {total_elements} elements"
    
    return result
