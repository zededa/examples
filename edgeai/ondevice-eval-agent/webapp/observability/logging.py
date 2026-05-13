"""Logging utilities for endpoint and processing step tracking."""

import os
import threading
from collections import deque
from datetime import datetime
from typing import Optional

# Global stores for real-time logs
endpoint_logs: Optional[deque] = None
processing_logs: Optional[deque] = None

# Thread locks for safe concurrent access to log queues
endpoint_logs_lock = threading.Lock()
processing_logs_lock = threading.Lock()


def init_log_queues(max_entries: Optional[int] = None) -> None:
    """Initialize log queues with configured max length."""
    global endpoint_logs, processing_logs
    if max_entries is None:
        max_entries = int(os.environ.get('MAX_LOG_ENTRIES', '100'))
    endpoint_logs = deque(maxlen=max_entries)
    processing_logs = deque(maxlen=max_entries)


def log_endpoint_call(
    endpoint: str,
    method: str,
    status_code: int,
    response_time: Optional[float] = None
) -> None:
    """Log endpoint calls for monitoring (thread-safe)."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'endpoint': endpoint,
        'method': method,
        'status': status_code,
        'response_time': response_time
    }
    with endpoint_logs_lock:
        if endpoint_logs is not None:
            endpoint_logs.append(log_entry)


def log_processing_step(step: str, details: str, status: str = "info") -> None:
    """Log processing steps for monitoring (thread-safe)."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'step': step,
        'details': details,
        'status': status
    }
    with processing_logs_lock:
        if processing_logs is not None:
            processing_logs.append(log_entry)


def clear_all_logs() -> None:
    """Clear all logs (thread-safe)."""
    with endpoint_logs_lock:
        if endpoint_logs is not None:
            endpoint_logs.clear()
    with processing_logs_lock:
        if processing_logs is not None:
            processing_logs.clear()


# Initialize on module load
init_log_queues()
