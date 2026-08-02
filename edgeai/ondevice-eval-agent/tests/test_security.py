"""
Tests for webapp/utils/secure_storage.py.

Covers Encryptor, StoredCredential, SecureStorage, and EncryptionError.
All SecureStorage tests use tmp_path with a fixed master_password to
avoid machine-id dependency.
"""

import json
from pathlib import Path

import pytest

from storage import (
    Encryptor,
    EncryptionError,
    SecureStorage,
    StoredCredential,
)
from storage.encryption import _derive_key


# ============================================================================
# Helpers
# ============================================================================

def _make_encryptor(password: str = "test-password") -> Encryptor:
    """Create an Encryptor with a deterministic key for tests."""
    salt = b"0123456789abcdef"
    key = _derive_key(password.encode(), salt)
    return Encryptor(key)


def _make_storage(tmp_path, master_password="test") -> SecureStorage:
    """Create a SecureStorage rooted under tmp_path."""
    return SecureStorage(
        storage_dir=str(tmp_path / "secure"),
        master_password=master_password,
    )


def _sample_credential(**overrides) -> StoredCredential:
    defaults = dict(
        name="test-provider",
        provider_type="openai-compatible",
        api_key="sk-secret-key-123",
        url="http://localhost:11434",
        model="llama3",
    )
    defaults.update(overrides)
    return StoredCredential(**defaults)


# ============================================================================
# Encryptor
# ============================================================================


class TestEncryptor:
    def test_encrypt_decrypt_roundtrip(self):
        enc = _make_encryptor()
        plaintext = "hello world"
        encrypted = enc.encrypt(plaintext)
        assert enc.decrypt(encrypted) == plaintext

    def test_decrypt_with_wrong_key_raises(self):
        enc_a = _make_encryptor("password-a")
        enc_b = _make_encryptor("password-b")
        encrypted = enc_a.encrypt("secret")
        with pytest.raises(EncryptionError):
            enc_b.decrypt(encrypted)

    def test_unicode_roundtrip(self):
        enc = _make_encryptor()
        text = "unicode test"
        assert enc.decrypt(enc.encrypt(text)) == text


# ============================================================================
# StoredCredential
# ============================================================================


class TestStoredCredential:
    def test_defaults(self):
        cred = StoredCredential(name="x", provider_type="openai")
        assert cred.enabled is True
        assert cred.priority == 10

    def test_url_normalization(self):
        cred = StoredCredential(
            name="x", provider_type="openai", url="localhost:1234"
        )
        assert cred.url == "http://localhost:1234"

    def test_url_normalization_preserves_https(self):
        cred = StoredCredential(
            name="x", provider_type="openai", url="https://api.example.com"
        )
        assert cred.url == "https://api.example.com"

    def test_to_dict_without_key(self):
        cred = _sample_credential()
        d = cred.to_dict(include_key=False)
        assert "api_key" not in d
        assert d["has_api_key"] is True

    def test_to_dict_with_key(self):
        cred = _sample_credential()
        d = cred.to_dict(include_key=True)
        assert d["api_key"] == "sk-secret-key-123"
        assert d["has_api_key"] is True

    def test_from_dict_roundtrip(self):
        original = _sample_credential()
        d = original.to_dict(include_key=True)
        restored = StoredCredential.from_dict(d)
        assert restored.name == original.name
        assert restored.api_key == original.api_key
        assert restored.provider_type == original.provider_type


# ============================================================================
# SecureStorage
# ============================================================================


class TestSecureStorage:
    def test_save_and_get_credential(self, tmp_path):
        storage = _make_storage(tmp_path)
        cred = _sample_credential()
        storage.save_credential(cred)
        retrieved = storage.get_credential("test-provider")
        assert retrieved is not None
        assert retrieved.api_key == "sk-secret-key-123"

    def test_get_nonexistent_returns_none(self, tmp_path):
        storage = _make_storage(tmp_path)
        assert storage.get_credential("does-not-exist") is None

    def test_delete_credential(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential())
        assert storage.delete_credential("test-provider") is True
        assert storage.get_credential("test-provider") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        storage = _make_storage(tmp_path)
        assert storage.delete_credential("nope") is False

    def test_list_credentials(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential(name="a"))
        storage.save_credential(_sample_credential(name="b"))
        creds = storage.list_credentials()
        assert isinstance(creds, list)
        assert len(creds) == 2

    def test_has_credential(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential())
        assert storage.has_credential("test-provider") is True
        assert storage.has_credential("missing") is False

    def test_get_all_enabled(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential(name="on", enabled=True))
        storage.save_credential(_sample_credential(name="off", enabled=False))
        enabled = storage.get_all_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "on"

    def test_export_and_import_credentials(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential(name="exp1"))
        exported = storage.export_credentials(include_keys=True)

        storage2 = _make_storage(tmp_path / "other")
        result = storage2.import_credentials(exported, overwrite=True)
        assert result["imported"] == 1
        assert storage2.has_credential("exp1")

    def test_import_no_overwrite_skips_existing(self, tmp_path):
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential(name="dup"))
        exported = storage.export_credentials(include_keys=True)

        result = storage.import_credentials(exported, overwrite=False)
        assert result["skipped"] == 1
        assert result["imported"] == 0

    def test_persistence_across_instances(self, tmp_path):
        """A new SecureStorage instance with the same dir should see saved data."""
        storage_dir = str(tmp_path / "persist")
        s1 = SecureStorage(storage_dir=storage_dir, master_password="test")
        s1.save_credential(_sample_credential())

        s2 = SecureStorage(storage_dir=storage_dir, master_password="test")
        assert s2.get_credential("test-provider") is not None
        assert s2.get_credential("test-provider").api_key == "sk-secret-key-123"

    def test_api_key_not_visible_in_raw_file(self, tmp_path):
        """The plaintext API key must NOT appear in the encrypted file on disk."""
        storage = _make_storage(tmp_path)
        storage.save_credential(_sample_credential())

        enc_file = Path(str(tmp_path / "secure")) / "credentials.enc"
        assert enc_file.exists()
        raw = enc_file.read_bytes()
        assert b"sk-secret-key-123" not in raw

    def test_created_at_has_timezone_info(self, tmp_path):
        """Bug fix #12: created_at must include UTC timezone information."""
        cred = _sample_credential()
        # created_at is set in __post_init__ via datetime.now(timezone.utc).isoformat()
        ts = cred.created_at
        # isoformat with timezone.utc produces "+00:00" suffix
        assert "+" in ts or "T" in ts, f"Timestamp missing timezone info: {ts}"
        # Specifically check for UTC offset
        assert "+00:00" in ts, f"Timestamp not UTC: {ts}"
