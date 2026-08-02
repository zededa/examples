"""
HTTP session management for Model Server Client.

This module handles HTTP session creation with retry logic and connection pooling.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import MAX_RETRIES, RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)


def create_session(
    max_retries: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
) -> requests.Session:
    """
    Create a requests session with retry logic and connection pooling.
    
    Args:
        max_retries: Maximum number of retry attempts for failed requests
        backoff_factor: Exponential backoff factor between retries
        
    Returns:
        Configured requests.Session instance
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET', 'HEAD', 'OPTIONS'],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session


class SessionManager:
    """
    Manages HTTP sessions with context manager support.
    
    Example:
        with SessionManager() as session:
            response = session.get("http://example.com")
    """
    
    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        backoff_factor: float = RETRY_BACKOFF_FACTOR,
    ) -> None:
        """
        Initialize the session manager.
        
        Args:
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff factor
        """
        self._session: Optional[requests.Session] = None
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
    
    @property
    def session(self) -> requests.Session:
        """Get or create the HTTP session."""
        if self._session is None:
            self._session = create_session(
                self._max_retries,
                self._backoff_factor,
            )
        return self._session
    
    def close(self) -> None:
        """Close the HTTP session and release resources."""
        if self._session is not None:
            self._session.close()
            self._session = None
            logger.debug("HTTP session closed")
    
    def __enter__(self) -> requests.Session:
        """Context manager entry - returns session."""
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes session."""
        self.close()


__all__ = [
    "create_session",
    "SessionManager",
]
