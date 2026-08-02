"""
Session Storage and State Management for MCP.

Provides utilities for managing session-specific file storage,
including path generation, size limits, cleanup, and session state tracking.

This module integrates with session_tracking.py for usage metrics and
warning state management, and with session_config.py for configuration.

Session Lifecycle:
    1. Session created via get_or_create_session()
    2. Usage tracked via record_* methods
    3. Warnings checked via check_session_warnings()
    4. Inactivity detected via check_inactivity_warnings()
    5. Cleanup after warning grace period via cleanup_inactive_sessions()
"""

from __future__ import annotations

import os
import shutil
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from .tracking import (
    SessionState,
    SessionUsageMetrics,
    SessionWarningState,
    UsageWarning,
    InactivityWarning,
    InactivityState,
    UsageDimension,
    WarningLevel,
    check_usage_warnings,
    check_inactivity_warning,
)
from .config import (
    get_session_config,
    get_usage_limits,
    get_inactivity_config,
    SessionConfig,
)

logger = logging.getLogger(__name__)

# Default storage configuration (also available via session_config)
SESSION_STORAGE_ROOT = os.environ.get('SESSION_STORAGE_ROOT', '/tmp/agent_sessions')
SESSION_STORAGE_LIMIT_MB = float(os.environ.get('SESSION_STORAGE_LIMIT_MB', '30'))


# =============================================================================
# Session Registry
# =============================================================================

class SessionRegistry:
    """
    Thread-safe registry for managing active sessions.
    
    Maintains all active SessionState objects and provides methods for
    session lifecycle management, usage tracking, and warning checks.
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID, or None if not found."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str) -> Tuple[SessionState, bool]:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Tuple of (SessionState, created) where created is True if new
        """
        config = get_session_config()
        
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id], False
            
            # Check capacity
            if len(self._sessions) >= config.max_concurrent_sessions:
                # Try to clean up expired sessions first
                self._cleanup_expired_no_lock(config)
                
                if len(self._sessions) >= config.max_concurrent_sessions:
                    raise SessionCapacityError(
                        f"Maximum concurrent sessions ({config.max_concurrent_sessions}) reached"
                    )
            
            session = SessionState(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            return session, True
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session from the registry."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Removed session from registry: {session_id}")
                return True
            return False
    
    def get_all_sessions(self) -> List[SessionState]:
        """Get all active sessions."""
        with self._lock:
            return list(self._sessions.values())
    
    def get_session_count(self) -> int:
        """Get count of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def _cleanup_expired_no_lock(self, config: SessionConfig) -> List[str]:
        """
        Internal cleanup without lock (caller must hold lock).
        
        Only cleans up sessions that have been warned and grace period expired.
        """
        inactivity_config = config.get_inactivity_config()
        expired = []
        
        for session_id, session in list(self._sessions.items()):
            if session.should_cleanup(inactivity_config):
                expired.append(session_id)
                del self._sessions[session_id]
        
        return expired
    
    def cleanup_expired_sessions(self) -> List[str]:
        """
        Clean up sessions that have been warned and grace period expired.
        
        Returns:
            List of session IDs that were cleaned up
        """
        config = get_session_config()
        
        with self._lock:
            expired = self._cleanup_expired_no_lock(config)
        
        # Cleanup storage for expired sessions (outside lock)
        for session_id in expired:
            cleanup_session_storage(session_id)
            logger.info(f"Cleaned up expired session: {session_id}")
        
        return expired


class SessionCapacityError(Exception):
    """Raised when session capacity is exceeded."""
    pass


# Global session registry instance
_session_registry = SessionRegistry()


def get_session_registry() -> SessionRegistry:
    """Get the global session registry."""
    return _session_registry


# =============================================================================
# Session Storage Functions (original API preserved)
# =============================================================================

def get_session_storage_path(session_id: str) -> str:
    """
    Get the storage directory path for a session.
    
    Creates the directory if it doesn't exist.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Absolute path to the session's storage directory
    """
    # Sanitize session_id to prevent path traversal
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('_', '-'))
    session_dir = os.path.join(SESSION_STORAGE_ROOT, safe_session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def get_session_storage_size_mb(session_id: str) -> float:
    """
    Calculate the total storage size for a session in MB.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Storage size in megabytes
    """
    session_dir = get_session_storage_path(session_id)
    
    total_size = 0
    if os.path.exists(session_dir):
        for dirpath, dirnames, filenames in os.walk(session_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass  # File may have been deleted
    
    return total_size / (1024 * 1024)


def count_session_images(session_id: str) -> int:
    """
    Count image files in session storage.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Number of image files
    """
    session_dir = get_session_storage_path(session_id)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    count = 0
    if os.path.exists(session_dir):
        for dirpath, dirnames, filenames in os.walk(session_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    count += 1
    
    return count


def check_session_storage_limit(session_id: str) -> Tuple[bool, float]:
    """
    Check if session storage is within the configured limit.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Tuple of (within_limit: bool, current_size_mb: float)
    """
    current_mb = get_session_storage_size_mb(session_id)
    config = get_session_config()
    within_limit = current_mb < config.max_storage_mb
    
    return within_limit, current_mb


def cleanup_session_storage(session_id: str) -> bool:
    """
    Clean up all storage for a session.
    
    Removes the session directory and all its contents.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if cleanup succeeded, False otherwise
    """
    # Sanitize session_id to prevent path traversal
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('_', '-'))
    session_dir = os.path.join(SESSION_STORAGE_ROOT, safe_session_id)
    
    try:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session storage: {session_dir}")
            return True
        return True  # Already clean
    except Exception as e:
        logger.error(f"Failed to cleanup session storage {session_dir}: {e}")
        return False


# =============================================================================
# Session State Management Functions
# =============================================================================

def get_or_create_session(session_id: str) -> Tuple[SessionState, bool]:
    """
    Get existing session or create new one.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Tuple of (SessionState, created) where created is True if new
    
    Raises:
        SessionCapacityError: If maximum concurrent sessions reached
    """
    return _session_registry.get_or_create_session(session_id)


def get_session(session_id: str) -> Optional[SessionState]:
    """Get a session by ID, or None if not found."""
    return _session_registry.get_session(session_id)


def remove_session(session_id: str, cleanup_storage: bool = True) -> bool:
    """
    Remove a session and optionally clean up its storage.
    
    Args:
        session_id: Unique session identifier
        cleanup_storage: Whether to also remove session storage
    
    Returns:
        True if session was removed
    """
    removed = _session_registry.remove_session(session_id)
    if removed and cleanup_storage:
        cleanup_session_storage(session_id)
    return removed


# =============================================================================
# Warning Check Functions
# =============================================================================

def check_session_warnings(session_id: str) -> Tuple[List[UsageWarning], Optional[InactivityWarning]]:
    """
    Check all warnings for a session.
    
    Checks both usage limits and inactivity state.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Tuple of (usage_warnings, inactivity_warning)
    """
    session = _session_registry.get_session(session_id)
    if session is None:
        return [], None
    
    config = get_session_config()
    limits = config.get_usage_limits()
    inactivity_config = config.get_inactivity_config()
    
    # Get current storage for storage limit check
    storage_mb = get_session_storage_size_mb(session_id)
    
    with session._lock:
        # Check usage warnings
        usage_warnings = check_usage_warnings(
            session.metrics,
            session.warning_state,
            limits,
            storage_mb=storage_mb,
        )
        
        # Check inactivity warning
        inactivity_warning = check_inactivity_warning(
            session.metrics,
            session.warning_state,
            inactivity_config,
        )
    
    return usage_warnings, inactivity_warning


def is_session_over_hard_limit(session_id: str) -> Tuple[bool, Optional[UsageDimension]]:
    """
    Check if session has exceeded any hard limit.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Tuple of (exceeded: bool, dimension: UsageDimension or None)
    """
    session = _session_registry.get_session(session_id)
    if session is None:
        return False, None
    
    config = get_session_config()
    limits = config.get_usage_limits()
    storage_mb = get_session_storage_size_mb(session_id)
    
    # Check each dimension
    dimension_values = {
        UsageDimension.TOKENS: session.metrics.total_tokens,
        UsageDimension.IMAGES: session.metrics.image_count,
        UsageDimension.REQUESTS: session.metrics.request_count,
        UsageDimension.STORAGE_MB: storage_mb,
    }
    
    for dimension, limit_config in limits.items():
        if not limit_config.enabled:
            continue
        current = dimension_values.get(dimension, 0)
        if current >= limit_config.hard_limit:
            return True, dimension
    
    return False, None


def cleanup_inactive_sessions() -> Tuple[List[str], List[InactivityWarning]]:
    """
    Check all sessions for inactivity and cleanup those ready for cleanup.
    
    This function implements the warning-before-cleanup flow:
    1. Identifies sessions that need inactivity warnings
    2. Returns warnings for sessions that should be notified
    3. Cleans up sessions that have been warned and grace period expired
    
    Returns:
        Tuple of (cleaned_up_session_ids, pending_warnings)
    """
    config = get_session_config()
    inactivity_config = config.get_inactivity_config()
    
    pending_warnings = []
    sessions_to_warn = []
    
    # First pass: check all sessions for warnings needed
    for session in _session_registry.get_all_sessions():
        with session._lock:
            warning = check_inactivity_warning(
                session.metrics,
                session.warning_state,
                inactivity_config,
            )
            if warning and warning.requires_response:
                pending_warnings.append(warning)
                sessions_to_warn.append(session.session_id)
    
    # Clean up sessions ready for cleanup
    cleaned_up = _session_registry.cleanup_expired_sessions()
    
    return cleaned_up, pending_warnings


def get_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive status for a session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Dictionary with session status, or None if session not found
    """
    session = _session_registry.get_session(session_id)
    if session is None:
        return None
    
    storage_mb = get_session_storage_size_mb(session_id)
    image_count = count_session_images(session_id)
    
    # Sync image count from storage
    with session._lock:
        if session.metrics.image_count < image_count:
            session.metrics.image_count = image_count
    
    status = session.to_dict()
    status['storage_mb'] = storage_mb
    status['storage_image_count'] = image_count
    
    return status
