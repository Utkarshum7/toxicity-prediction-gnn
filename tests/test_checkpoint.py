import hashlib
import io

import pytest

from src import checkpoint

PAYLOAD = b"not-a-real-checkpoint-but-fine-for-hash-tests"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("TOX_MODEL_PATH", "TOX_MODEL_URL", "TOX_MODEL_SHA256"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(checkpoint.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_local_file_is_used_and_verified(monkeypatch, tmp_path):
    path = tmp_path / "m.pth"
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("TOX_MODEL_PATH", str(path))
    monkeypatch.setenv("TOX_MODEL_SHA256", SHA.upper())  # case-insensitive
    assert checkpoint.resolve_checkpoint() == path


def test_bom_and_whitespace_in_checksum_are_tolerated(monkeypatch, tmp_path):
    path = tmp_path / "m.pth"
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("TOX_MODEL_PATH", str(path))
    monkeypatch.setenv("TOX_MODEL_SHA256", "﻿  " + SHA + "\r\n")  # what a PowerShell pipe produced
    assert checkpoint.resolve_checkpoint() == path


@pytest.mark.parametrize("bad", ["abc123", "z" * 64, SHA + "00"])
def test_malformed_checksum_is_rejected_clearly(monkeypatch, tmp_path, bad):
    path = tmp_path / "m.pth"
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("TOX_MODEL_PATH", str(path))
    monkeypatch.setenv("TOX_MODEL_SHA256", bad)
    with pytest.raises(checkpoint.CheckpointError, match="64-character"):
        checkpoint.resolve_checkpoint()


def test_checksum_mismatch_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "m.pth"
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("TOX_MODEL_PATH", str(path))
    monkeypatch.setenv("TOX_MODEL_SHA256", "0" * 64)
    with pytest.raises(checkpoint.CheckpointError, match="mismatch"):
        checkpoint.resolve_checkpoint()


def test_missing_checkpoint_without_url_has_helpful_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TOX_MODEL_PATH", str(tmp_path / "absent.pth"))
    with pytest.raises(checkpoint.CheckpointError, match="TOX_MODEL_URL"):
        checkpoint.resolve_checkpoint()


def test_download_is_verified_cached_and_not_repeated(monkeypatch, tmp_path):
    monkeypatch.setenv("TOX_MODEL_PATH", str(tmp_path / "absent.pth"))
    monkeypatch.setenv("TOX_MODEL_URL", "https://example.invalid/model.pth")
    monkeypatch.setenv("TOX_MODEL_SHA256", SHA)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", fake_urlopen)
    first = checkpoint.resolve_checkpoint()
    second = checkpoint.resolve_checkpoint()
    assert first == second and first.read_bytes() == PAYLOAD
    assert len(calls) == 1, "checkpoint must be downloaded once per instance, not per request"
    assert not list(first.parent.glob("*.part"))


def test_corrupt_download_is_rejected_and_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("TOX_MODEL_PATH", str(tmp_path / "absent.pth"))
    monkeypatch.setenv("TOX_MODEL_URL", "https://example.invalid/model.pth")
    monkeypatch.setenv("TOX_MODEL_SHA256", SHA)
    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"tampered"))
    with pytest.raises(checkpoint.CheckpointError, match="SHA-256"):
        checkpoint.resolve_checkpoint()
    assert not list((tmp_path / "tmp").glob("*"))


def test_non_https_url_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("TOX_MODEL_PATH", str(tmp_path / "absent.pth"))
    monkeypatch.setenv("TOX_MODEL_URL", "http://example.invalid/model.pth")
    with pytest.raises(checkpoint.CheckpointError, match="https"):
        checkpoint.resolve_checkpoint()


def test_download_errors_do_not_leak_the_url(monkeypatch, tmp_path):
    secret_url = "https://example.invalid/model.pth?token=SUPERSECRET"
    monkeypatch.setenv("TOX_MODEL_PATH", str(tmp_path / "absent.pth"))
    monkeypatch.setenv("TOX_MODEL_URL", secret_url)

    def fail(url, timeout):
        raise OSError(f"connection to {url} failed")

    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", fail)
    with pytest.raises(checkpoint.CheckpointError) as err:
        checkpoint.resolve_checkpoint()
    assert "SUPERSECRET" not in str(err.value)
