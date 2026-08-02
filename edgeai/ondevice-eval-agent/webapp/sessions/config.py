"""
Session Configuration for MCP Sessions.

This module provides centralized, configurable settings for session management
including usage limits, warning thresholds, and inactivity timeouts.

All values can be overridden via environment variables for deployment flexibility.

Environment Variables:
    SESSION_MAX_TOKENS: Maximum tokens per session (default: 100000)
    SESSION_MAX_IMAGES: Maximum images per session (default: 50)
    SESSION_MAX_REQUESTS: Maximum requests per session (default: 500)
    SESSION_STORAGE_LIMIT_MB: Maximum storage per session in MB (default: 30)
    
    SESSION_SOFT_WARNING_RATIO: Ratio for soft warning (default: 0.8)
    SESSION_CRITICAL_WARNING_RATIO: Ratio for critical warning (default: 0.95)
    
    SESSION_IDLE_THRESHOLD_MINUTES: Minutes before idle (default: 30)
    SESSION_WARNING_THRESHOLD_MINUTES: Minutes before inactivity warning (default: 50)
    SESSION_GRACE_PERIOD_MINUTES: Minutes grace after warning (default: 10)

Usage:
    from sessions.config import get_session_config, SessionConfig
    
    config = get_session_config()
    limits = config.get_usage_limits()
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from .tracking import (
    UsageDimension,
    UsageLimitConfig,
    InactivityConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def _get_env_float(name: str, default: float) -> float:
    """Get a float from environment variable with default."""
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {name}, using default: {default}")
        return default


def _get_env_int(name: str, default: int) -> int:
    """Get an integer from environment variable with default."""
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {name}, using default: {default}")
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Get a boolean from environment variable with default."""
    val = os.environ.get(name, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


# =============================================================================
# Session Configuration
# =============================================================================

@dataclass
class SessionConfig:
    """
    Complete configuration for session management.
    
    Attributes:
        max_tokens: Maximum tokens allowed per session
        max_images: Maximum image assets allowed per session
        max_requests: Maximum requests allowed per session
        max_storage_mb: Maximum storage in MB per session
        
        soft_warning_ratio: Percentage of limit for soft warning (0-1)
        critical_warning_ratio: Percentage of limit for critical warning (0-1)
        
        enable_token_limits: Whether to enforce token limits
        enable_image_limits: Whether to enforce image limits
        enable_request_limits: Whether to enforce request limits
        enable_storage_limits: Whether to enforce storage limits
        
        idle_threshold_seconds: Seconds before session is considered idle
        warning_threshold_seconds: Seconds before inactivity warning
        grace_period_seconds: Seconds after warning before cleanup
        enable_inactivity_warnings: Whether to enable inactivity warnings
        
        max_concurrent_sessions: Maximum concurrent sessions per server
        max_history_length: Maximum conversation history length
        max_tool_calls: Maximum tool calls to retain per session
    """
    # Usage limits
    max_tokens: int = 100_000
    max_images: int = 50
    max_requests: int = 500
    max_storage_mb: float = 30.0
    
    # Warning thresholds (as ratios of hard limit)
    soft_warning_ratio: float = 0.8
    critical_warning_ratio: float = 0.95
    
    # Limit enforcement toggles
    enable_token_limits: bool = True
    enable_image_limits: bool = True
    enable_request_limits: bool = True
    enable_storage_limits: bool = True
    
    # Inactivity settings (in seconds)
    idle_threshold_seconds: float = 1800.0      # 30 minutes
    warning_threshold_seconds: float = 3000.0   # 50 minutes
    grace_period_seconds: float = 600.0         # 10 minutes
    enable_inactivity_warnings: bool = True
    
    # Session management
    max_concurrent_sessions: int = 1000
    max_history_length: int = 20
    max_tool_calls: int = 50
    
    def get_usage_limits(self) -> Dict[UsageDimension, UsageLimitConfig]:
        """Build usage limit configurations from settings."""
        return {
            UsageDimension.TOKENS: UsageLimitConfig(
                dimension=UsageDimension.TOKENS,
                hard_limit=float(self.max_tokens),
                soft_warning_ratio=self.soft_warning_ratio,
                critical_warning_ratio=self.critical_warning_ratio,
                enabled=self.enable_token_limits,
            ),
            UsageDimension.IMAGES: UsageLimitConfig(
                dimension=UsageDimension.IMAGES,
                hard_limit=float(self.max_images),
                soft_warning_ratio=self.soft_warning_ratio,
                critical_warning_ratio=self.critical_warning_ratio,
                enabled=self.enable_image_limits,
            ),
            UsageDimension.REQUESTS: UsageLimitConfig(
                dimension=UsageDimension.REQUESTS,
                hard_limit=float(self.max_requests),
                soft_warning_ratio=self.soft_warning_ratio,
                critical_warning_ratio=self.critical_warning_ratio,
                enabled=self.enable_request_limits,
            ),
            UsageDimension.STORAGE_MB: UsageLimitConfig(
                dimension=UsageDimension.STORAGE_MB,
                hard_limit=self.max_storage_mb,
                soft_warning_ratio=self.soft_warning_ratio,
                critical_warning_ratio=self.critical_warning_ratio,
                enabled=self.enable_storage_limits,
            ),
        }
    
    def get_inactivity_config(self) -> InactivityConfig:
        """Build inactivity configuration from settings."""
        return InactivityConfig(
            idle_threshold_seconds=self.idle_threshold_seconds,
            warning_threshold_seconds=self.warning_threshold_seconds,
            grace_period_seconds=self.grace_period_seconds,
            enabled=self.enable_inactivity_warnings,
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for inspection."""
        return {
            "limits": {
                "max_tokens": self.max_tokens,
                "max_images": self.max_images,
                "max_requests": self.max_requests,
                "max_storage_mb": self.max_storage_mb,
            },
            "warnings": {
                "soft_warning_ratio": self.soft_warning_ratio,
                "critical_warning_ratio": self.critical_warning_ratio,
            },
            "enforcement": {
                "enable_token_limits": self.enable_token_limits,
                "enable_image_limits": self.enable_image_limits,
                "enable_request_limits": self.enable_request_limits,
                "enable_storage_limits": self.enable_storage_limits,
            },
            "inactivity": {
                "idle_threshold_seconds": self.idle_threshold_seconds,
                "warning_threshold_seconds": self.warning_threshold_seconds,
                "grace_period_seconds": self.grace_period_seconds,
                "enable_inactivity_warnings": self.enable_inactivity_warnings,
            },
            "session": {
                "max_concurrent_sessions": self.max_concurrent_sessions,
                "max_history_length": self.max_history_length,
                "max_tool_calls": self.max_tool_calls,
            },
        }


# =============================================================================
# Configuration Loading
# =============================================================================

_config_instance: Optional[SessionConfig] = None
_config_lock = threading.Lock()


def load_session_config() -> SessionConfig:
    """
    Load session configuration from environment variables.
    
    Returns:
        SessionConfig instance with values from environment or defaults
    """
    return SessionConfig(
        # Usage limits
        max_tokens=_get_env_int('SESSION_MAX_TOKENS', 100_000),
        max_images=_get_env_int('SESSION_MAX_IMAGES', 50),
        max_requests=_get_env_int('SESSION_MAX_REQUESTS', 500),
        max_storage_mb=_get_env_float('SESSION_STORAGE_LIMIT_MB', 30.0),
        
        # Warning thresholds
        soft_warning_ratio=_get_env_float('SESSION_SOFT_WARNING_RATIO', 0.8),
        critical_warning_ratio=_get_env_float('SESSION_CRITICAL_WARNING_RATIO', 0.95),
        
        # Limit enforcement toggles
        enable_token_limits=_get_env_bool('SESSION_ENABLE_TOKEN_LIMITS', True),
        enable_image_limits=_get_env_bool('SESSION_ENABLE_IMAGE_LIMITS', True),
        enable_request_limits=_get_env_bool('SESSION_ENABLE_REQUEST_LIMITS', True),
        enable_storage_limits=_get_env_bool('SESSION_ENABLE_STORAGE_LIMITS', True),
        
        # Inactivity settings (convert from minutes in env to seconds)
        idle_threshold_seconds=_get_env_float('SESSION_IDLE_THRESHOLD_MINUTES', 30) * 60,
        warning_threshold_seconds=_get_env_float('SESSION_WARNING_THRESHOLD_MINUTES', 50) * 60,
        grace_period_seconds=_get_env_float('SESSION_GRACE_PERIOD_MINUTES', 10) * 60,
        enable_inactivity_warnings=_get_env_bool('SESSION_ENABLE_INACTIVITY_WARNINGS', True),
        
        # Session management
        max_concurrent_sessions=_get_env_int('MAX_AGENT_SESSIONS', 1000),
        max_history_length=_get_env_int('SESSION_MAX_HISTORY_LENGTH', 20),
        max_tool_calls=_get_env_int('SESSION_MAX_TOOL_CALLS', 50),
    )


def get_session_config() -> SessionConfig:
    """
    Get the global session configuration instance.

    Lazily loads configuration on first access. Thread-safe.

    Returns:
        SessionConfig singleton instance
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = load_session_config()
                logger.info(f"Loaded session configuration: {_config_instance.to_dict()}")
    return _config_instance


def reload_session_config() -> SessionConfig:
    """
    Force reload of session configuration from environment.

    Useful for testing or dynamic reconfiguration.

    Returns:
        New SessionConfig instance
    """
    global _config_instance
    with _config_lock:
        _config_instance = load_session_config()
        logger.info(f"Reloaded session configuration: {_config_instance.to_dict()}")
    return _config_instance


# =============================================================================
# Convenience Accessors
# =============================================================================

def get_usage_limits() -> Dict[UsageDimension, UsageLimitConfig]:
    """Get usage limit configurations."""
    return get_session_config().get_usage_limits()


def get_inactivity_config() -> InactivityConfig:
    """Get inactivity configuration."""
    return get_session_config().get_inactivity_config()
