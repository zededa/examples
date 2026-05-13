"""
Encrypted local storage for LLM credentials and other sensitive data.

Structure:
    encryption.py   - Fernet Encryptor + PBKDF2 key derivation + machine-id
    credentials.py  - StoredCredential + SecureStorage manager + singleton helpers

Usage:
    from storage import get_secure_storage, StoredCredential
    storage = get_secure_storage()
    storage.save_credential(StoredCredential(name="anthropic", provider_type="anthropic", api_key="..."))
"""

from .encryption import (
    EncryptionError,
    Encryptor,
)

from .credentials import (
    StoredCredential,
    SecureStorage,
    get_secure_storage,
    reset_secure_storage,
)

__all__ = [
    "EncryptionError",
    "Encryptor",
    "StoredCredential",
    "SecureStorage",
    "get_secure_storage",
    "reset_secure_storage",
]
