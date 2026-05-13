"""
Encryption primitives for the credentials storage.

Fernet symmetric encryption (AES-128-CBC + HMAC via the cryptography
package), PBKDF2 key derivation from a machine-derived password and salt.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Final

_KEY_ITERATIONS: Final[int] = 100_000


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""
    pass


def _get_machine_id() -> str:
    """
    Build a stable machine identifier by hashing hostname + machine type
    + processor info, plus /etc/machine-id when available.
    """
    components = [
        platform.node(),
        platform.machine(),
        platform.processor(),
    ]

    try:
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            components.append(machine_id_path.read_text().strip())
    except (OSError, PermissionError):
        pass

    combined = "|".join(filter(None, components))
    return hashlib.sha256(combined.encode()).hexdigest()


def _derive_key(password: bytes, salt: bytes) -> bytes:
    """
    PBKDF2-derive a 32-byte Fernet-compatible key from password and salt.
    """
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as e:
        raise RuntimeError(
            "cryptography library is required for secure storage. "
            "Install with: pip install cryptography"
        ) from e

    import base64

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_KEY_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password))


class Encryptor:
    """
    Handles encryption and decryption of string data via Fernet.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key

        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise RuntimeError(
                "cryptography library is required for secure storage. "
                "Install with: pip install cryptography"
            ) from e

        self._fernet = Fernet(key)

    def encrypt(self, data: str) -> bytes:
        return self._fernet.encrypt(data.encode("utf-8"))

    def decrypt(self, encrypted_data: bytes) -> str:
        try:
            return self._fernet.decrypt(encrypted_data).decode("utf-8")
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}") from e


__all__ = [
    "EncryptionError",
    "Encryptor",
    "_get_machine_id",
    "_derive_key",
]
