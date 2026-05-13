"""
Tests for Session Tracking and Warning System.

These tests verify the session management features including:
- Per-session usage tracking
- Soft limits with early warnings
- Inactivity warning flow
- Cleanup after warning

Run with: pytest tests/test_session_tracking.py -v
"""

import os
import sys
import time
import tempfile
import shutil
from unittest import mock

import pytest

# Add webapp to path for imports
webapp_path = os.path.join(os.path.dirname(__file__), '..', 'webapp')
sys.path.insert(0, webapp_path)

from sessions.tracking import (
    SessionUsageMetrics,
    SessionWarningState,
    SessionState,
    UsageDimension,
    WarningLevel,
    InactivityState,
    UsageLimitConfig,
    InactivityConfig,
    UsageWarning,
    InactivityWarning,
    check_usage_warnings,
    check_inactivity_warning,
)

from sessions.config import (
    SessionConfig,
    load_session_config,
    get_session_config,
    reload_session_config,
)


# =============================================================================
# SessionUsageMetrics Tests
# =============================================================================

class TestSessionUsageMetrics:
    """Tests for SessionUsageMetrics dataclass."""
    
    def test_initial_state(self):
        """Test that metrics initialize with zero values."""
        metrics = SessionUsageMetrics()
        
        assert metrics.total_tokens == 0
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.image_count == 0
        assert metrics.request_count == 0
        assert metrics.tool_call_count == 0
        assert metrics.created_at > 0
        assert metrics.last_activity > 0
    
    def test_record_tokens(self):
        """Test token recording accumulates correctly."""
        metrics = SessionUsageMetrics()
        
        metrics.record_tokens(prompt=100, completion=50)
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        
        metrics.record_tokens(prompt=200, completion=100)
        assert metrics.prompt_tokens == 300
        assert metrics.completion_tokens == 150
        assert metrics.total_tokens == 450
    
    def test_record_image(self):
        """Test image recording."""
        metrics = SessionUsageMetrics()
        
        metrics.record_image()
        assert metrics.image_count == 1
        
        metrics.record_image(count=5)
        assert metrics.image_count == 6
    
    def test_record_request(self):
        """Test request recording."""
        metrics = SessionUsageMetrics()
        
        metrics.record_request()
        assert metrics.request_count == 1
        assert metrics.last_request_at is not None
        
        metrics.record_request()
        assert metrics.request_count == 2
    
    def test_touch_updates_activity(self):
        """Test that touch() updates last_activity timestamp."""
        metrics = SessionUsageMetrics()
        initial_activity = metrics.last_activity
        
        time.sleep(0.01)  # Small delay
        metrics.touch()
        
        assert metrics.last_activity > initial_activity
    
    def test_inactivity_calculation(self):
        """Test inactivity seconds calculation."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time() - 60  # 60 seconds ago
        
        inactivity = metrics.get_inactivity_seconds()
        assert 59 < inactivity < 61
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = SessionUsageMetrics()
        metrics.record_tokens(100, 50)
        metrics.record_image(2)
        
        data = metrics.to_dict()
        
        assert data["total_tokens"] == 150
        assert data["image_count"] == 2
        assert "created_at" in data
        assert "last_activity" in data
        assert "inactivity_seconds" in data


# =============================================================================
# UsageLimitConfig Tests
# =============================================================================

class TestUsageLimitConfig:
    """Tests for UsageLimitConfig."""
    
    def test_threshold_calculation(self):
        """Test soft and critical threshold calculations."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
            soft_warning_ratio=0.8,
            critical_warning_ratio=0.95,
        )
        
        assert config.soft_threshold == 800
        assert config.critical_threshold == 950
    
    def test_warning_level_none(self):
        """Test no warning when under soft threshold."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
            soft_warning_ratio=0.8,
        )
        
        assert config.get_warning_level(700) == WarningLevel.NONE
    
    def test_warning_level_soft(self):
        """Test soft warning level."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
            soft_warning_ratio=0.8,
            critical_warning_ratio=0.95,
        )
        
        assert config.get_warning_level(850) == WarningLevel.SOFT
    
    def test_warning_level_hard(self):
        """Test hard/critical warning level."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
            soft_warning_ratio=0.8,
            critical_warning_ratio=0.95,
        )
        
        assert config.get_warning_level(960) == WarningLevel.HARD
    
    def test_warning_level_exceeded(self):
        """Test exceeded warning level."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
        )
        
        assert config.get_warning_level(1000) == WarningLevel.EXCEEDED
        assert config.get_warning_level(1100) == WarningLevel.EXCEEDED
    
    def test_disabled_limit(self):
        """Test that disabled limits don't trigger warnings."""
        config = UsageLimitConfig(
            dimension=UsageDimension.TOKENS,
            hard_limit=1000,
            enabled=False,
        )
        
        assert config.get_warning_level(2000) == WarningLevel.NONE


# =============================================================================
# SessionWarningState Tests
# =============================================================================

class TestSessionWarningState:
    """Tests for SessionWarningState."""
    
    def test_initial_state(self):
        """Test initial warning state."""
        state = SessionWarningState()
        
        assert state.inactivity_state == InactivityState.ACTIVE
        assert len(state.usage_warnings_issued) == 0
        assert state.inactivity_warning_sent_at is None
    
    def test_should_issue_warning_new(self):
        """Test that new warnings should be issued."""
        state = SessionWarningState()
        
        assert state.should_issue_warning(UsageDimension.TOKENS, WarningLevel.SOFT)
        assert state.should_issue_warning(UsageDimension.TOKENS, WarningLevel.HARD)
    
    def test_should_not_reissue_same_warning(self):
        """Test that same warning level is not re-issued."""
        state = SessionWarningState()
        
        state.record_warning_issued(UsageDimension.TOKENS, WarningLevel.SOFT)
        
        assert not state.should_issue_warning(UsageDimension.TOKENS, WarningLevel.SOFT)
    
    def test_should_issue_higher_warning(self):
        """Test that higher severity warnings are issued after lower."""
        state = SessionWarningState()
        
        state.record_warning_issued(UsageDimension.TOKENS, WarningLevel.SOFT)
        
        assert state.should_issue_warning(UsageDimension.TOKENS, WarningLevel.HARD)
        assert state.should_issue_warning(UsageDimension.TOKENS, WarningLevel.EXCEEDED)
    
    def test_inactivity_warning_tracking(self):
        """Test inactivity warning state transitions."""
        state = SessionWarningState()
        
        state.mark_inactivity_warning_sent()
        
        assert state.inactivity_state == InactivityState.WARNING_SENT
        assert state.inactivity_warning_sent_at is not None
    
    def test_grace_period_calculation(self):
        """Test grace period remaining calculation."""
        state = SessionWarningState()
        state.inactivity_warning_sent_at = time.time() - 30  # 30 seconds ago
        
        remaining = state.get_grace_period_remaining(60)  # 60 second grace
        assert 29 < remaining < 31
    
    def test_grace_period_expired(self):
        """Test grace period expiration check."""
        state = SessionWarningState()
        state.inactivity_warning_sent_at = time.time() - 120  # 2 minutes ago
        
        assert state.is_grace_period_expired(60)  # 60 second grace
    
    def test_reset_inactivity_state(self):
        """Test resetting inactivity state when user responds."""
        state = SessionWarningState()
        state.inactivity_state = InactivityState.WARNING_SENT
        state.inactivity_warning_sent_at = time.time()
        
        state.reset_inactivity_state()
        
        assert state.inactivity_state == InactivityState.ACTIVE
        assert state.inactivity_warning_sent_at is None


# =============================================================================
# check_usage_warnings Tests
# =============================================================================

class TestCheckUsageWarnings:
    """Tests for check_usage_warnings function."""
    
    def test_no_warnings_under_threshold(self):
        """Test no warnings when under all thresholds."""
        metrics = SessionUsageMetrics()
        metrics.total_tokens = 500
        
        warning_state = SessionWarningState()
        
        limits = {
            UsageDimension.TOKENS: UsageLimitConfig(
                dimension=UsageDimension.TOKENS,
                hard_limit=10000,
            ),
        }
        
        warnings = check_usage_warnings(metrics, warning_state, limits)
        
        assert len(warnings) == 0
    
    def test_soft_warning_issued(self):
        """Test soft warning is issued at threshold."""
        metrics = SessionUsageMetrics()
        metrics.total_tokens = 8500  # 85% of 10000
        
        warning_state = SessionWarningState()
        
        limits = {
            UsageDimension.TOKENS: UsageLimitConfig(
                dimension=UsageDimension.TOKENS,
                hard_limit=10000,
                soft_warning_ratio=0.8,
            ),
        }
        
        warnings = check_usage_warnings(metrics, warning_state, limits)
        
        assert len(warnings) == 1
        assert warnings[0].dimension == UsageDimension.TOKENS
        assert warnings[0].level == WarningLevel.SOFT
    
    def test_multiple_dimension_warnings(self):
        """Test warnings across multiple dimensions."""
        metrics = SessionUsageMetrics()
        metrics.total_tokens = 8500
        metrics.image_count = 45
        
        warning_state = SessionWarningState()
        
        limits = {
            UsageDimension.TOKENS: UsageLimitConfig(
                dimension=UsageDimension.TOKENS,
                hard_limit=10000,
                soft_warning_ratio=0.8,
            ),
            UsageDimension.IMAGES: UsageLimitConfig(
                dimension=UsageDimension.IMAGES,
                hard_limit=50,
                soft_warning_ratio=0.8,
            ),
        }
        
        warnings = check_usage_warnings(metrics, warning_state, limits)
        
        assert len(warnings) == 2
    
    def test_warning_not_reissued(self):
        """Test that warnings are not re-issued."""
        metrics = SessionUsageMetrics()
        metrics.total_tokens = 8500
        
        warning_state = SessionWarningState()
        warning_state.record_warning_issued(UsageDimension.TOKENS, WarningLevel.SOFT)
        
        limits = {
            UsageDimension.TOKENS: UsageLimitConfig(
                dimension=UsageDimension.TOKENS,
                hard_limit=10000,
                soft_warning_ratio=0.8,
            ),
        }
        
        warnings = check_usage_warnings(metrics, warning_state, limits)
        
        assert len(warnings) == 0


# =============================================================================
# check_inactivity_warning Tests
# =============================================================================

class TestCheckInactivityWarning:
    """Tests for check_inactivity_warning function."""
    
    def test_no_warning_when_active(self):
        """Test no warning when session is active."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time()  # Just now
        
        warning_state = SessionWarningState()
        
        config = InactivityConfig(
            idle_threshold_seconds=1800,
            warning_threshold_seconds=3000,
        )
        
        warning = check_inactivity_warning(metrics, warning_state, config)
        
        assert warning is None
    
    def test_transition_to_idle(self):
        """Test transition from active to idle."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time() - 2000  # Past idle threshold
        
        warning_state = SessionWarningState()
        
        config = InactivityConfig(
            idle_threshold_seconds=1800,
            warning_threshold_seconds=3000,
        )
        
        warning = check_inactivity_warning(metrics, warning_state, config)
        
        assert warning is None
        assert warning_state.inactivity_state == InactivityState.IDLE
    
    def test_warning_issued_at_threshold(self):
        """Test warning is issued at warning threshold."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time() - 3500  # Past warning threshold
        
        warning_state = SessionWarningState()
        warning_state.inactivity_state = InactivityState.IDLE
        
        config = InactivityConfig(
            idle_threshold_seconds=1800,
            warning_threshold_seconds=3000,
            grace_period_seconds=600,
        )
        
        # First call transitions to WARNING_PENDING
        check_inactivity_warning(metrics, warning_state, config)
        
        # Second call should issue the warning
        warning = check_inactivity_warning(metrics, warning_state, config)
        
        assert warning is not None
        assert warning.requires_response
        assert not warning.cleanup_imminent
    
    def test_cleanup_after_grace_period(self):
        """Test cleanup pending after grace period expires."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time() - 4000
        
        warning_state = SessionWarningState()
        warning_state.inactivity_state = InactivityState.WARNING_SENT
        warning_state.inactivity_warning_sent_at = time.time() - 700  # Grace period expired
        
        config = InactivityConfig(
            warning_threshold_seconds=3000,
            grace_period_seconds=600,
        )
        
        warning = check_inactivity_warning(metrics, warning_state, config)
        
        assert warning is not None
        assert warning.cleanup_imminent
        assert warning_state.inactivity_state == InactivityState.CLEANUP_PENDING
    
    def test_disabled_inactivity_handling(self):
        """Test no warnings when inactivity handling is disabled."""
        metrics = SessionUsageMetrics()
        metrics.last_activity = time.time() - 10000  # Very inactive
        
        warning_state = SessionWarningState()
        
        config = InactivityConfig(
            warning_threshold_seconds=3000,
            enabled=False,
        )
        
        warning = check_inactivity_warning(metrics, warning_state, config)
        
        assert warning is None


# =============================================================================
# SessionState Tests
# =============================================================================

class TestSessionState:
    """Tests for SessionState class."""
    
    def test_creation(self):
        """Test session state creation."""
        session = SessionState(session_id="test-123")
        
        assert session.session_id == "test-123"
        assert session.metrics is not None
        assert session.warning_state is not None
        assert session.exploration_context == "initial"
    
    def test_touch_resets_inactivity_warning(self):
        """Test that user activity resets inactivity warnings."""
        session = SessionState(session_id="test-123")
        session.warning_state.inactivity_state = InactivityState.WARNING_SENT
        session.warning_state.inactivity_warning_sent_at = time.time()
        
        session.touch()
        
        assert session.warning_state.inactivity_state == InactivityState.ACTIVE
        assert session.warning_state.inactivity_warning_sent_at is None
    
    def test_record_operations(self):
        """Test various recording operations."""
        session = SessionState(session_id="test-123")
        
        session.record_request()
        assert session.metrics.request_count == 1
        
        session.record_tokens(100, 50)
        assert session.metrics.total_tokens == 150
        
        session.record_image()
        assert session.metrics.image_count == 1
    
    def test_history_management(self):
        """Test conversation history with size limits."""
        session = SessionState(session_id="test-123")
        
        for i in range(25):
            session.add_history("user", f"Message {i}")
        
        assert len(session.history) == 20  # Limited to max_history
    
    def test_to_dict(self):
        """Test serialization."""
        session = SessionState(session_id="test-123")
        session.record_tokens(100, 50)
        
        data = session.to_dict()
        
        assert data["session_id"] == "test-123"
        assert "metrics" in data
        assert "warning_state" in data


# =============================================================================
# SessionConfig Tests
# =============================================================================

class TestSessionConfig:
    """Tests for SessionConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SessionConfig()
        
        assert config.max_tokens == 100_000
        assert config.max_images == 50
        assert config.soft_warning_ratio == 0.8
        assert config.enable_inactivity_warnings
    
    def test_get_usage_limits(self):
        """Test building usage limit configs."""
        config = SessionConfig(
            max_tokens=50000,
            soft_warning_ratio=0.75,
        )
        
        limits = config.get_usage_limits()
        
        assert UsageDimension.TOKENS in limits
        assert limits[UsageDimension.TOKENS].hard_limit == 50000
        assert limits[UsageDimension.TOKENS].soft_warning_ratio == 0.75
    
    def test_get_inactivity_config(self):
        """Test building inactivity config."""
        config = SessionConfig(
            idle_threshold_seconds=900,
            warning_threshold_seconds=1500,
            grace_period_seconds=300,
        )
        
        inactivity = config.get_inactivity_config()
        
        assert inactivity.idle_threshold_seconds == 900
        assert inactivity.warning_threshold_seconds == 1500
        assert inactivity.grace_period_seconds == 300
    
    def test_load_from_environment(self):
        """Test loading config from environment variables."""
        with mock.patch.dict(os.environ, {
            'SESSION_MAX_TOKENS': '50000',
            'SESSION_SOFT_WARNING_RATIO': '0.7',
            'SESSION_IDLE_THRESHOLD_MINUTES': '15',
        }):
            config = load_session_config()
            
            assert config.max_tokens == 50000
            assert config.soft_warning_ratio == 0.7
            assert config.idle_threshold_seconds == 900  # 15 * 60


# =============================================================================
# Integration Tests
# =============================================================================

class TestSessionIntegration:
    """Integration tests for session management."""
    
    def test_full_warning_flow(self):
        """Test complete warning flow from creation to cleanup."""
        # Create session
        session = SessionState(session_id="integration-test")
        
        # Simulate high token usage
        session.record_tokens(85000, 0)  # 85% of 100k default
        
        # Check warnings
        config = SessionConfig()
        limits = config.get_usage_limits()
        
        warnings = check_usage_warnings(
            session.metrics,
            session.warning_state,
            limits,
        )
        
        # Should get soft warning
        assert len(warnings) == 1
        assert warnings[0].level == WarningLevel.SOFT
        
        # Continue using, hit critical
        session.record_tokens(11000, 0)  # Now at 96%
        
        warnings = check_usage_warnings(
            session.metrics,
            session.warning_state,
            limits,
        )
        
        # Should get hard warning (upgrade from soft)
        assert len(warnings) == 1
        assert warnings[0].level == WarningLevel.HARD
    
    def test_inactivity_warning_flow(self):
        """Test complete inactivity warning flow."""
        session = SessionState(session_id="inactivity-test")
        
        config = InactivityConfig(
            idle_threshold_seconds=10,
            warning_threshold_seconds=20,
            grace_period_seconds=10,
        )
        
        # Simulate time passing
        session.metrics.last_activity = time.time() - 25
        
        # Should transition through states
        check_inactivity_warning(session.metrics, session.warning_state, config)
        assert session.warning_state.inactivity_state == InactivityState.WARNING_PENDING
        
        warning = check_inactivity_warning(session.metrics, session.warning_state, config)
        assert warning is not None
        assert warning.requires_response
        
        # User responds
        session.touch()
        assert session.warning_state.inactivity_state == InactivityState.ACTIVE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
