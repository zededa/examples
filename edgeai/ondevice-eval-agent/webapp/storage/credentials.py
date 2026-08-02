"""
Encrypted credential storage.

StoredCredential dataclass and SecureStorage manager. Encryption primitives
(Fernet, PBKDF2 key derivation, machine-id) live in `storage.encryption`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

from .encryption import (
    EncryptionError,
    Encryptor,
    _derive_key,
    _get_machine_id,
)

logger = logging.getLogger(__name__)


_DEFAULT_STORAGE_DIR: Final[str] = ".edgeai"
_CREDENTIALS_FILE: Final[str] = "credentials.enc"
_SALT_FILE: Final[str] = ".salt"
_SALT_LENGTH: Final[int] = 16


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StoredCredential:
    """
    A stored credential entry.
    
    Attributes:
        name: Unique identifier for this credential.
        provider_type: Type of LLM provider.
        api_key: Encrypted API key (stored encrypted).
        url: Server URL (optional).
        model: Default model name.
        metadata: Additional configuration.
        created_at: Timestamp when created.
        updated_at: Timestamp when last updated.
    """
    name: str
    provider_type: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    model: Optional[str] = None
    priority: int = 10
    max_tokens: int = 4096
    temperature: float = 0.1
    enabled: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    metadata: Dict[str, Any] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        # Normalize URL to ensure it has http:// or https:// scheme
        if self.url:
            self.url = self._normalize_url(self.url)
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensure URL has proper http:// or https:// scheme."""
        url = url.strip()
        if not url:
            return url
        # If URL doesn't start with http:// or https://, add http://
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        # Remove trailing slashes for consistency
        return url.rstrip('/')
    
    def to_dict(self, include_key: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "provider_type": self.provider_type,
            "url": self.url,
            "model": self.model,
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enabled": self.enabled,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_api_key": bool(self.api_key),
        }
        if include_key:
            result["api_key"] = self.api_key
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredCredential":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            provider_type=data.get("provider_type", "openai-compatible"),
            api_key=data.get("api_key"),
            url=data.get("url"),
            model=data.get("model"),
            priority=data.get("priority", 10),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.1),
            enabled=data.get("enabled", True),
            supports_tools=data.get("supports_tools", True),
            supports_vision=data.get("supports_vision", False),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# =============================================================================
# Secure Storage Manager
# =============================================================================

class SecureStorage:
    """
    Manages encrypted storage of LLM credentials and configuration.
    
    Provides thread-safe access to encrypted credential storage with
    automatic key derivation based on machine identity.
    
    Example:
        >>> storage = SecureStorage()
        >>> storage.save_credential(StoredCredential(
        ...     name="openai",
        ...     provider_type="openai",
        ...     api_key="sk-..."
        ... ))
        >>> cred = storage.get_credential("openai")
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        master_password: Optional[str] = None,
    ) -> None:
        """
        Initialize secure storage.
        
        Args:
            storage_dir: Directory for storing encrypted files.
                        Defaults to ~/.edgeai/
            master_password: Optional master password for additional security.
                           If not provided, uses machine-derived key.
        """
        # Determine storage directory
        if storage_dir:
            self._storage_dir = Path(storage_dir)
        else:
            self._storage_dir = Path.home() / _DEFAULT_STORAGE_DIR
        
        # Ensure directory exists with restricted permissions
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._storage_dir, 0o700)
        except OSError:
            pass  # May fail on some systems
        
        # File paths
        self._credentials_path = self._storage_dir / _CREDENTIALS_FILE
        self._salt_path = self._storage_dir / _SALT_FILE
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize encryption
        self._encryptor = self._initialize_encryption(master_password)
        
        # Load existing credentials
        self._credentials: Dict[str, StoredCredential] = {}
        self._load_credentials()
        
        logger.info(f"Secure storage initialized at {self._storage_dir}")
    
    # =========================================================================
    # Public API - Credential Management
    # =========================================================================
    
    def save_credential(self, credential: StoredCredential) -> bool:
        """
        Save or update a credential.
        
        Args:
            credential: The credential to save.
            
        Returns:
            True if saved successfully.
        """
        with self._lock:
            # Update timestamp
            credential.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Store in memory
            self._credentials[credential.name] = credential
            
            # Persist to disk
            return self._save_credentials()
    
    def get_credential(self, name: str) -> Optional[StoredCredential]:
        """
        Get a credential by name.
        
        Args:
            name: Credential name.
            
        Returns:
            The credential or None if not found.
        """
        with self._lock:
            return self._credentials.get(name)
    
    def delete_credential(self, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            name: Credential name to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if name not in self._credentials:
                return False
            
            del self._credentials[name]
            return self._save_credentials()
    
    def list_credentials(self, include_keys: bool = False) -> List[Dict[str, Any]]:
        """
        List all stored credentials.
        
        Args:
            include_keys: Whether to include API keys in output.
                         WARNING: Setting True exposes plaintext secrets.
            
        Returns:
            List of credential dictionaries.
        """
        with self._lock:
            return [
                cred.to_dict(include_key=include_keys)
                for cred in self._credentials.values()
            ]
    
    def has_credential(self, name: str) -> bool:
        """Check if a credential exists."""
        with self._lock:
            return name in self._credentials
    
    def get_all_enabled(self) -> List[StoredCredential]:
        """Get all enabled credentials."""
        with self._lock:
            return [
                cred for cred in self._credentials.values()
                if cred.enabled
            ]
    
    # =========================================================================
    # Public API - Bulk Operations
    # =========================================================================
    
    def export_credentials(self, include_keys: bool = True) -> Dict[str, Any]:
        """
        Export all credentials for backup.
        
        Args:
            include_keys: Whether to include API keys.
            
        Returns:
            Dictionary with all credentials and metadata.
        """
        with self._lock:
            return {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "credentials": self.list_credentials(include_keys=include_keys),
            }
    
    def import_credentials(
        self,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """
        Import credentials from export data.
        
        Args:
            data: Exported credentials data.
            overwrite: Whether to overwrite existing credentials.
            
        Returns:
            Dictionary with counts: imported, skipped, errors.
        """
        results = {"imported": 0, "skipped": 0, "errors": 0}
        
        credentials = data.get("credentials", [])
        
        with self._lock:
            for cred_data in credentials:
                try:
                    name = cred_data.get("name")
                    if not name:
                        results["errors"] += 1
                        continue
                    
                    if name in self._credentials and not overwrite:
                        results["skipped"] += 1
                        continue
                    
                    credential = StoredCredential.from_dict(cred_data)
                    self._credentials[name] = credential
                    results["imported"] += 1
                    
                except Exception as e:
                    logger.error(f"Error importing credential: {e}")
                    results["errors"] += 1
            
            # Save all changes
            if results["imported"] > 0:
                self._save_credentials()
        
        return results
    
    # =========================================================================
    # Private - Encryption Setup
    # =========================================================================
    
    def _initialize_encryption(self, master_password: Optional[str]) -> Encryptor:
        """Initialize encryption with derived key."""
        # Get or create salt
        salt = self._get_or_create_salt()
        
        # Derive password from master password and/or machine ID
        if master_password:
            password = f"{master_password}:{_get_machine_id()}".encode()
        else:
            password = _get_machine_id().encode()
        
        # Derive encryption key
        key = _derive_key(password, salt)
        
        return Encryptor(key)
    
    def _get_or_create_salt(self) -> bytes:
        """
        Get existing salt or create a new one.
        
        Returns:
            Salt bytes for key derivation.
            
        Raises:
            RuntimeError: If salt file cannot be read or created.
        """
        try:
            if self._salt_path.exists():
                return self._salt_path.read_bytes()
            
            # Generate new salt
            salt = os.urandom(_SALT_LENGTH)
            self._salt_path.write_bytes(salt)
            
            # Restrict permissions
            try:
                os.chmod(self._salt_path, 0o600)
            except OSError:
                pass
            
            return salt
        except Exception as e:
            raise RuntimeError(f"Failed to manage salt file: {e}") from e
    
    # =========================================================================
    # Private - Persistence
    # =========================================================================
    
    def _load_credentials(self) -> None:
        """Load credentials from encrypted file."""
        if not self._credentials_path.exists():
            logger.debug("No existing credentials file found")
            return
        
        try:
            encrypted_data = self._credentials_path.read_bytes()
            decrypted_json = self._encryptor.decrypt(encrypted_data)
            data = json.loads(decrypted_json)
            
            for cred_data in data.get("credentials", []):
                try:
                    credential = StoredCredential.from_dict(cred_data)
                    self._credentials[credential.name] = credential
                except Exception as e:
                    logger.error(f"Error loading credential: {e}")
            
            logger.info(f"Loaded {len(self._credentials)} credentials")
            
        except EncryptionError as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            logger.warning("Credentials file may be corrupted or key changed")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid credentials file format: {e}")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
    
    def _save_credentials(self) -> bool:
        """Save credentials to encrypted file."""
        try:
            # Build data structure
            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "credentials": [
                    cred.to_dict(include_key=True)
                    for cred in self._credentials.values()
                ],
            }
            
            # Encrypt and save
            json_data = json.dumps(data, indent=2)
            encrypted_data = self._encryptor.encrypt(json_data)
            
            # Write atomically with restrictive permissions
            temp_path = self._credentials_path.with_suffix(".tmp")
            # Create with restricted permissions before writing content
            fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, encrypted_data)
            finally:
                os.close(fd)
            temp_path.replace(self._credentials_path)
            
            # Restrict permissions
            try:
                os.chmod(self._credentials_path, 0o600)
            except OSError:
                pass
            
            logger.debug(f"Saved {len(self._credentials)} credentials")
            return True
            
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False


# =============================================================================
# Module-Level Singleton
# =============================================================================

_storage_instance: Optional[SecureStorage] = None
_storage_lock = threading.Lock()


def get_secure_storage(
    storage_dir: Optional[str] = None,
    master_password: Optional[str] = None,
) -> SecureStorage:
    """
    Get the singleton SecureStorage instance.
    
    Args:
        storage_dir: Optional custom storage directory.
        master_password: Optional master password.
        
    Returns:
        SecureStorage instance.
    """
    global _storage_instance
    
    with _storage_lock:
        if _storage_instance is None:
            _storage_instance = SecureStorage(
                storage_dir=storage_dir,
                master_password=master_password,
            )
        return _storage_instance


def reset_secure_storage() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _storage_instance
    with _storage_lock:
        _storage_instance = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "SecureStorage",
    "StoredCredential",
    "EncryptionError",
    "Encryptor",
    "get_secure_storage",
    "reset_secure_storage",
]
