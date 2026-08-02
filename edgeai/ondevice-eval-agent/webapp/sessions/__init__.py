"""
Per-session lifecycle, usage metrics, and warning state.

Structure:
    registry.py  - SessionRegistry + storage utilities (was mcp/session.py)
    tracking.py  - SessionState, UsageMetrics, warning logic (was session_tracking.py)
    config.py    - SessionConfig, env-driven limits (was session_config.py)

Usage:
    from sessions.registry import get_or_create_session, check_session_warnings
    from sessions.tracking import SessionState, WarningLevel, UsageDimension
    from sessions.config  import get_session_config
"""

from .registry import (
    SessionRegistry,
    SessionCapacityError,
    get_session_registry,
    get_or_create_session,
    get_session,
    remove_session,
    check_session_warnings,
    is_session_over_hard_limit,
    cleanup_inactive_sessions,
    get_session_status,
    get_session_storage_size_mb,
    count_session_images,
    get_session_storage_path,
    check_session_storage_limit,
    cleanup_session_storage,
    SESSION_STORAGE_ROOT,
    SESSION_STORAGE_LIMIT_MB,
)

from .tracking import (
    SessionState,
    SessionUsageMetrics,
    SessionWarningState,
    UsageWarning,
    InactivityWarning,
    UsageDimension,
    WarningLevel,
    InactivityState,
    UsageLimitConfig,
    InactivityConfig,
    check_usage_warnings,
    check_inactivity_warning,
)

from .config import (
    SessionConfig,
    get_session_config,
    reload_session_config,
    load_session_config,
    get_usage_limits,
    get_inactivity_config,
)

__all__ = [
    # Registry
    "SessionRegistry",
    "SessionCapacityError",
    "get_session_registry",
    "get_or_create_session",
    "get_session",
    "remove_session",
    "check_session_warnings",
    "is_session_over_hard_limit",
    "cleanup_inactive_sessions",
    "get_session_status",
    "get_session_storage_size_mb",
    "count_session_images",
    "get_session_storage_path",
    "check_session_storage_limit",
    "cleanup_session_storage",
    "SESSION_STORAGE_ROOT",
    "SESSION_STORAGE_LIMIT_MB",
    # Tracking types
    "SessionState",
    "SessionUsageMetrics",
    "SessionWarningState",
    "UsageWarning",
    "InactivityWarning",
    "UsageDimension",
    "WarningLevel",
    "InactivityState",
    "UsageLimitConfig",
    "InactivityConfig",
    "check_usage_warnings",
    "check_inactivity_warning",
    # Config
    "SessionConfig",
    "get_session_config",
    "reload_session_config",
    "load_session_config",
    "get_usage_limits",
    "get_inactivity_config",
]
