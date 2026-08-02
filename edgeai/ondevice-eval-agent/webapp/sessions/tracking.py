"""
Session Usage Tracking and Soft Limits for MCP Sessions.

This module provides per-session usage tracking with configurable soft limits
and early warning capabilities. It tracks:
- Token consumption across all model calls
- Image asset creation
- Request counts
- Activity timestamps

Soft limits emit warnings before hard limits are reached, giving users
an opportunity to adjust their usage or acknowledge continued operation.

Usage:
    from sessions.tracking import (
        SessionUsageMetrics,
        SessionWarningState,
        UsageLimitConfig,
        check_usage_warnings,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class UsageDimension(Enum):
    """Dimensions of usage that can be tracked and limited."""
    TOKENS = "tokens"
    IMAGES = "images"
    REQUESTS = "requests"
    STORAGE_MB = "storage_mb"


class WarningLevel(Enum):
    """Warning severity levels for usage limits."""
    NONE = "none"
    SOFT = "soft"          # Approaching limit (e.g., 80%)
    HARD = "hard"          # At or near hard limit (e.g., 95%)
    EXCEEDED = "exceeded"  # Hard limit exceeded


class InactivityState(Enum):
    """States for session inactivity handling."""
    ACTIVE = "active"              # Session is actively being used
    IDLE = "idle"                  # No recent activity but within normal bounds
    WARNING_PENDING = "warning_pending"  # Inactivity warning should be sent
    WARNING_SENT = "warning_sent"  # User has been warned about pending cleanup
    CLEANUP_PENDING = "cleanup_pending"  # Grace period expired, cleanup imminent


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class UsageLimitConfig:
    """
    Configuration for a single usage limit with soft warning threshold.
    
    Attributes:
        dimension: The usage dimension this limit applies to
        hard_limit: The absolute maximum allowed value
        soft_warning_ratio: Ratio (0-1) of hard_limit at which to warn (e.g., 0.8 = 80%)
        critical_warning_ratio: Ratio (0-1) for critical warning (e.g., 0.95 = 95%)
        enabled: Whether this limit is actively enforced
    """
    dimension: UsageDimension
    hard_limit: float
    soft_warning_ratio: float = 0.8
    critical_warning_ratio: float = 0.95
    enabled: bool = True
    
    @property
    def soft_threshold(self) -> float:
        """Get the absolute soft warning threshold."""
        return self.hard_limit * self.soft_warning_ratio
    
    @property
    def critical_threshold(self) -> float:
        """Get the absolute critical warning threshold."""
        return self.hard_limit * self.critical_warning_ratio
    
    def get_warning_level(self, current_value: float) -> WarningLevel:
        """Determine warning level for a given usage value."""
        if not self.enabled:
            return WarningLevel.NONE
        if current_value >= self.hard_limit:
            return WarningLevel.EXCEEDED
        if current_value >= self.critical_threshold:
            return WarningLevel.HARD
        if current_value >= self.soft_threshold:
            return WarningLevel.SOFT
        return WarningLevel.NONE
    
    def get_usage_percentage(self, current_value: float) -> float:
        """Get current usage as percentage of hard limit."""
        if self.hard_limit <= 0:
            return 0.0
        return min(100.0, (current_value / self.hard_limit) * 100)


@dataclass
class InactivityConfig:
    """
    Configuration for session inactivity handling.
    
    Attributes:
        idle_threshold_seconds: Time before session is considered idle
        warning_threshold_seconds: Time after which inactivity warning is sent
        grace_period_seconds: Time after warning before cleanup occurs
        enabled: Whether inactivity handling is active
    """
    idle_threshold_seconds: float = 1800.0    # 30 minutes
    warning_threshold_seconds: float = 3000.0  # 50 minutes
    grace_period_seconds: float = 600.0        # 10 minutes after warning
    enabled: bool = True
    
    @property
    def total_timeout_seconds(self) -> float:
        """Total time from last activity to cleanup."""
        return self.warning_threshold_seconds + self.grace_period_seconds


# =============================================================================
# Session Usage Metrics
# =============================================================================

@dataclass
class SessionUsageMetrics:
    """
    Cumulative usage metrics for a single session.
    
    All metrics are session-scoped and persist for the session lifetime.
    Thread-safe through external locking in SessionState.
    
    Attributes:
        total_tokens: Cumulative tokens consumed across all model calls
        prompt_tokens: Cumulative prompt/input tokens
        completion_tokens: Cumulative completion/output tokens
        image_count: Number of image assets created
        request_count: Total number of requests (all types, not just tools)
        tool_call_count: Number of tool invocations
        created_at: Session creation timestamp
        last_activity: Most recent activity timestamp
        last_request_at: Timestamp of last request
    """
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_count: int = 0
    request_count: int = 0
    tool_call_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_request_at: Optional[float] = None
    
    def record_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Record token usage from a model call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.touch()
    
    def record_image(self, count: int = 1) -> None:
        """Record image asset creation."""
        self.image_count += count
        self.touch()
    
    def record_request(self) -> None:
        """Record a request (any type)."""
        self.request_count += 1
        self.last_request_at = time.time()
        self.touch()
    
    def record_tool_call(self) -> None:
        """Record a tool invocation."""
        self.tool_call_count += 1
        self.touch()
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def get_inactivity_seconds(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self.last_activity
    
    def get_session_duration_seconds(self) -> float:
        """Get total session duration in seconds."""
        return time.time() - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "image_count": self.image_count,
            "request_count": self.request_count,
            "tool_call_count": self.tool_call_count,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "last_request_at": self.last_request_at,
            "inactivity_seconds": self.get_inactivity_seconds(),
            "session_duration_seconds": self.get_session_duration_seconds(),
        }


# =============================================================================
# Session Warning State
# =============================================================================

@dataclass
class SessionWarningState:
    """
    Tracks warning state for a session across all dimensions.
    
    Maintains which warnings have been issued to avoid duplicate warnings
    and tracks the inactivity warning flow state.
    
    Attributes:
        usage_warnings_issued: Map of dimension -> highest warning level issued
        inactivity_state: Current state in inactivity warning flow
        inactivity_warning_sent_at: When inactivity warning was sent
        warnings_acknowledged: Whether user has acknowledged current warnings
    """
    usage_warnings_issued: Dict[UsageDimension, WarningLevel] = field(default_factory=dict)
    inactivity_state: InactivityState = InactivityState.ACTIVE
    inactivity_warning_sent_at: Optional[float] = None
    warnings_acknowledged: bool = False
    
    def should_issue_warning(
        self, 
        dimension: UsageDimension, 
        level: WarningLevel
    ) -> bool:
        """Check if a warning should be issued (hasn't been issued yet)."""
        if level == WarningLevel.NONE:
            return False
        
        current_level = self.usage_warnings_issued.get(dimension, WarningLevel.NONE)
        
        # Issue warning if this is a higher severity than previously issued
        level_order = [WarningLevel.NONE, WarningLevel.SOFT, WarningLevel.HARD, WarningLevel.EXCEEDED]
        return level_order.index(level) > level_order.index(current_level)
    
    def record_warning_issued(
        self, 
        dimension: UsageDimension, 
        level: WarningLevel
    ) -> None:
        """Record that a warning has been issued."""
        self.usage_warnings_issued[dimension] = level
    
    def mark_inactivity_warning_sent(self) -> None:
        """Mark that an inactivity warning has been sent."""
        self.inactivity_state = InactivityState.WARNING_SENT
        self.inactivity_warning_sent_at = time.time()
    
    def get_grace_period_remaining(self, grace_period_seconds: float) -> float:
        """Get remaining time in grace period after warning."""
        if self.inactivity_warning_sent_at is None:
            return grace_period_seconds
        elapsed = time.time() - self.inactivity_warning_sent_at
        return max(0, grace_period_seconds - elapsed)
    
    def is_grace_period_expired(self, grace_period_seconds: float) -> bool:
        """Check if the grace period after warning has expired."""
        return self.get_grace_period_remaining(grace_period_seconds) <= 0
    
    def reset_inactivity_state(self) -> None:
        """Reset inactivity state (called when user responds)."""
        self.inactivity_state = InactivityState.ACTIVE
        self.inactivity_warning_sent_at = None
    
    def acknowledge_warnings(self) -> None:
        """Mark that user has acknowledged current warnings."""
        self.warnings_acknowledged = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert warning state to dictionary for serialization."""
        return {
            "usage_warnings_issued": {
                k.value: v.value for k, v in self.usage_warnings_issued.items()
            },
            "inactivity_state": self.inactivity_state.value,
            "inactivity_warning_sent_at": self.inactivity_warning_sent_at,
            "warnings_acknowledged": self.warnings_acknowledged,
        }


# =============================================================================
# Usage Warning Checker
# =============================================================================

@dataclass
class UsageWarning:
    """A single usage warning to be communicated to the user."""
    dimension: UsageDimension
    level: WarningLevel
    current_value: float
    limit_value: float
    percentage: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "level": self.level.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "percentage": self.percentage,
            "message": self.message,
        }


def check_usage_warnings(
    metrics: SessionUsageMetrics,
    warning_state: SessionWarningState,
    limits: Dict[UsageDimension, UsageLimitConfig],
    storage_mb: float = 0.0,
) -> List[UsageWarning]:
    """
    Check all usage dimensions for warnings.
    
    Args:
        metrics: Current session usage metrics
        warning_state: Current warning state for the session
        limits: Map of dimension to limit configuration
        storage_mb: Current storage usage in MB (calculated externally)
    
    Returns:
        List of new warnings that should be issued
    """
    warnings = []
    
    # Map dimensions to their current values
    dimension_values = {
        UsageDimension.TOKENS: metrics.total_tokens,
        UsageDimension.IMAGES: metrics.image_count,
        UsageDimension.REQUESTS: metrics.request_count,
        UsageDimension.STORAGE_MB: storage_mb,
    }
    
    for dimension, config in limits.items():
        if not config.enabled:
            continue
        
        current_value = dimension_values.get(dimension, 0)
        level = config.get_warning_level(current_value)
        
        if warning_state.should_issue_warning(dimension, level):
            percentage = config.get_usage_percentage(current_value)
            message = _build_warning_message(dimension, level, current_value, config, percentage)
            
            warnings.append(UsageWarning(
                dimension=dimension,
                level=level,
                current_value=current_value,
                limit_value=config.hard_limit,
                percentage=percentage,
                message=message,
            ))
            
            # Record that this warning has been issued
            warning_state.record_warning_issued(dimension, level)
    
    return warnings


def _build_warning_message(
    dimension: UsageDimension,
    level: WarningLevel,
    current: float,
    config: UsageLimitConfig,
    percentage: float,
) -> str:
    """Build a human-readable warning message."""
    dimension_names = {
        UsageDimension.TOKENS: "token usage",
        UsageDimension.IMAGES: "image assets",
        UsageDimension.REQUESTS: "requests",
        UsageDimension.STORAGE_MB: "storage",
    }
    
    dimension_name = dimension_names.get(dimension, dimension.value)
    
    if dimension == UsageDimension.STORAGE_MB:
        current_str = f"{current:.1f}MB"
        limit_str = f"{config.hard_limit:.1f}MB"
    else:
        current_str = f"{int(current):,}"
        limit_str = f"{int(config.hard_limit):,}"
    
    if level == WarningLevel.SOFT:
        return (
            f"⚠️ Your session {dimension_name} is approaching the limit "
            f"({percentage:.0f}% used: {current_str} / {limit_str}). "
            f"Consider wrapping up or starting a new session."
        )
    elif level == WarningLevel.HARD:
        return (
            f"🚨 Your session {dimension_name} is near the limit "
            f"({percentage:.0f}% used: {current_str} / {limit_str}). "
            f"You may experience restrictions soon."
        )
    elif level == WarningLevel.EXCEEDED:
        return (
            f"❌ Your session {dimension_name} has exceeded the limit "
            f"({current_str} / {limit_str}). "
            f"Please start a new session to continue."
        )
    
    return ""


# =============================================================================
# Inactivity Warning Flow
# =============================================================================

@dataclass
class InactivityWarning:
    """An inactivity warning to be communicated to the user."""
    state: InactivityState
    inactivity_seconds: float
    grace_remaining_seconds: float
    message: str
    requires_response: bool
    cleanup_imminent: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "inactivity_seconds": self.inactivity_seconds,
            "grace_remaining_seconds": self.grace_remaining_seconds,
            "message": self.message,
            "requires_response": self.requires_response,
            "cleanup_imminent": self.cleanup_imminent,
        }


def check_inactivity_warning(
    metrics: SessionUsageMetrics,
    warning_state: SessionWarningState,
    config: InactivityConfig,
) -> Optional[InactivityWarning]:
    """
    Check if an inactivity warning should be issued.
    
    Implements the warning flow:
    1. ACTIVE -> IDLE (user can continue, no warning)
    2. IDLE -> WARNING_PENDING (time to warn user)
    3. WARNING_PENDING -> WARNING_SENT (warning issued)
    4. WARNING_SENT -> CLEANUP_PENDING (grace period expired)
    
    Args:
        metrics: Current session usage metrics
        warning_state: Current warning state for the session
        config: Inactivity configuration
    
    Returns:
        InactivityWarning if user should be warned, None otherwise
    """
    if not config.enabled:
        return None
    
    inactivity = metrics.get_inactivity_seconds()
    current_state = warning_state.inactivity_state
    
    # State transitions
    if current_state == InactivityState.ACTIVE:
        if inactivity >= config.warning_threshold_seconds:
            warning_state.inactivity_state = InactivityState.WARNING_PENDING
        elif inactivity >= config.idle_threshold_seconds:
            warning_state.inactivity_state = InactivityState.IDLE
        return None
    
    elif current_state == InactivityState.IDLE:
        if inactivity >= config.warning_threshold_seconds:
            warning_state.inactivity_state = InactivityState.WARNING_PENDING
        elif inactivity < config.idle_threshold_seconds:
            # User became active again
            warning_state.reset_inactivity_state()
        return None
    
    elif current_state == InactivityState.WARNING_PENDING:
        # Issue the warning
        warning_state.mark_inactivity_warning_sent()
        
        return InactivityWarning(
            state=InactivityState.WARNING_SENT,
            inactivity_seconds=inactivity,
            grace_remaining_seconds=config.grace_period_seconds,
            message=_build_inactivity_warning_message(inactivity, config.grace_period_seconds),
            requires_response=True,
            cleanup_imminent=False,
        )
    
    elif current_state == InactivityState.WARNING_SENT:
        grace_remaining = warning_state.get_grace_period_remaining(config.grace_period_seconds)
        
        if grace_remaining <= 0:
            warning_state.inactivity_state = InactivityState.CLEANUP_PENDING
            return InactivityWarning(
                state=InactivityState.CLEANUP_PENDING,
                inactivity_seconds=inactivity,
                grace_remaining_seconds=0,
                message=_build_cleanup_imminent_message(),
                requires_response=True,
                cleanup_imminent=True,
            )
        
        # User hasn't responded but grace period not yet expired
        return None
    
    elif current_state == InactivityState.CLEANUP_PENDING:
        # Cleanup should proceed
        return InactivityWarning(
            state=InactivityState.CLEANUP_PENDING,
            inactivity_seconds=inactivity,
            grace_remaining_seconds=0,
            message=_build_cleanup_imminent_message(),
            requires_response=False,
            cleanup_imminent=True,
        )
    
    return None


def _build_inactivity_warning_message(inactivity_seconds: float, grace_seconds: float) -> str:
    """Build a human-readable inactivity warning message."""
    inactive_mins = int(inactivity_seconds / 60)
    grace_mins = int(grace_seconds / 60)
    
    return (
        f"⏰ Your session has been inactive for {inactive_mins} minutes. "
        f"To keep your session alive and preserve your chat history and uploaded images, "
        f"please respond within the next {grace_mins} minutes. "
        f"If no response is received, your session will be cleaned up automatically."
    )


def _build_cleanup_imminent_message() -> str:
    """Build a message indicating cleanup is about to occur."""
    return (
        "⚠️ Session cleanup is imminent. Your session has been inactive too long "
        "and the grace period has expired. Your session data will be cleaned up. "
        "Send any message now to keep your session alive, or start a new session."
    )


# =============================================================================
# Session State Manager
# =============================================================================

@dataclass
class SessionState:
    """
    Complete state for a tracked session.
    
    Combines usage metrics, warning state, and session-specific data.
    Thread-safe through explicit locking.
    
    Attributes:
        session_id: Unique session identifier
        metrics: Usage metrics for this session
        warning_state: Warning tracking state
        history: Conversation history
        tool_calls: List of tool call records
        current_model: Currently selected model
        exploration_context: Context for recommendations
        _lock: Thread lock for safe concurrent access
    """
    session_id: str
    metrics: SessionUsageMetrics = field(default_factory=SessionUsageMetrics)
    warning_state: SessionWarningState = field(default_factory=SessionWarningState)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    current_model: Optional[str] = None
    exploration_context: str = "initial"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def touch(self) -> None:
        """Update activity timestamp and reset inactivity warnings if appropriate."""
        with self._lock:
            self.metrics.touch()
            # If user responds, reset inactivity warning state
            if self.warning_state.inactivity_state in (
                InactivityState.WARNING_SENT,
                InactivityState.CLEANUP_PENDING,
            ):
                self.warning_state.reset_inactivity_state()
                logger.info(f"Session {self.session_id}: User activity reset inactivity warning")
    
    def record_request(self) -> None:
        """Record a new request."""
        with self._lock:
            self.metrics.record_request()
    
    def record_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Record token usage."""
        with self._lock:
            self.metrics.record_tokens(prompt, completion)
    
    def record_image(self, count: int = 1) -> None:
        """Record image creation."""
        with self._lock:
            self.metrics.record_image(count)
    
    def record_tool_call(self) -> None:
        """Record a tool invocation."""
        with self._lock:
            self.metrics.record_tool_call()
    
    def add_history(self, role: str, content: str, max_history: int = 20) -> None:
        """Add to conversation history with size limit."""
        with self._lock:
            self.history.append({"role": role, "content": content})
            if len(self.history) > max_history:
                self.history = self.history[-max_history:]
    
    def add_tool_call(self, tool_call: Dict[str, Any], max_tools: int = 50) -> None:
        """Add a tool call record with size limit."""
        with self._lock:
            self.tool_calls.append(tool_call)
            self.metrics.record_tool_call()
            if len(self.tool_calls) > max_tools:
                self.tool_calls = self.tool_calls[-max_tools:]
    
    def get_context(self) -> Dict[str, Any]:
        """Get session context for response enrichment."""
        with self._lock:
            return {
                "exploration_state": self.exploration_context,
                "current_model": self.current_model,
                "tools_used_count": len(self.tool_calls),
                "metrics": self.metrics.to_dict(),
                "warning_state": self.warning_state.to_dict(),
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert full session state to dictionary."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "metrics": self.metrics.to_dict(),
                "warning_state": self.warning_state.to_dict(),
                "current_model": self.current_model,
                "exploration_context": self.exploration_context,
                "history_length": len(self.history),
                "tool_calls_count": len(self.tool_calls),
            }
    
    def should_cleanup(self, config: InactivityConfig) -> bool:
        """Check if session should be cleaned up based on warning flow."""
        with self._lock:
            # Only cleanup if warning was sent AND grace period expired
            if self.warning_state.inactivity_state == InactivityState.CLEANUP_PENDING:
                return True
            
            # Legacy fallback: cleanup if total timeout exceeded and no tracking
            total_inactive = self.metrics.get_inactivity_seconds()
            return total_inactive >= config.total_timeout_seconds
