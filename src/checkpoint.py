"""Locate (and if necessary fetch) the trained model checkpoint.

Resolution order:
1. ``TOX_MODEL_PATH`` or ``models/model.pth`` if the file exists (local runs, or a checkpoint
   bundled into a Vercel deployment with ``vercel deploy`` from a machine that has it).
2. ``TOX_MODEL_URL`` (https only): downloaded once per instance into the temp directory
   (``/tmp`` on serverless platforms, the only writable location) and reused afterwards.

If ``TOX_MODEL_SHA256`` is set, the file's SHA-256 must match or loading is refused. The
checkpoint is never modified.
"""
import hashlib
import logging
import os
import tempfile
import threading
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_TIMEOUT = 60  # seconds
MAX_BYTES = 200 * 1024 * 1024  # refuse implausibly large downloads
_lock = threading.Lock()


class CheckpointError(RuntimeError):
    """Checkpoint missing, unreachable or failing verification."""


def default_path():
    return Path(os.environ.get("TOX_MODEL_PATH") or PROJECT_ROOT / "models" / "model.pth")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256():
    """Normalised TOX_MODEL_SHA256 (or None). Tolerates stray whitespace / a UTF-8 BOM that
    shells sometimes prepend when piping a value, but rejects anything that is not a SHA-256."""
    raw = os.environ.get("TOX_MODEL_SHA256")
    if raw is None:
        return None
    value = raw.replace("﻿", "").strip().lower()
    if not value:
        return None
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CheckpointError("TOX_MODEL_SHA256 must be a 64-character hexadecimal SHA-256 digest")
    return value


def _verify(path, expected):
    if not expected:
        return
    actual = sha256_of(path)
    if actual != expected:
        raise CheckpointError(f"Checkpoint SHA-256 mismatch for {path.name} (expected {expected[:12]}…, got {actual[:12]}…)")


def _download(url, destination, expected):
    if urlparse(url).scheme != "https":
        raise CheckpointError("TOX_MODEL_URL must be an https URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response, open(partial, "wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                size += len(block)
                if size > MAX_BYTES:
                    raise CheckpointError("Checkpoint download exceeds the size limit")
                digest.update(block)
                out.write(block)
        if expected and digest.hexdigest() != expected:
            raise CheckpointError("Downloaded checkpoint failed SHA-256 verification")
        os.replace(partial, destination)  # atomic: readers never see a half-written file
    except CheckpointError:
        raise
    except Exception as exc:  # network errors etc. (message deliberately omits the URL)
        raise CheckpointError(f"Could not download checkpoint ({type(exc).__name__})") from exc
    finally:
        partial.unlink(missing_ok=True)


def resolve_checkpoint():
    """Return a verified local path to the checkpoint, downloading it if configured."""
    expected = expected_sha256()
    local = default_path()
    with _lock:
        if local.is_file():
            _verify(local, expected)
            return local
        url = os.environ.get("TOX_MODEL_URL")
        if not url:
            raise CheckpointError(
                f"Model checkpoint not found at {local}. Train one with `python train.py`, "
                "bundle it into the deployment, or set TOX_MODEL_URL."
            )
        cached = Path(tempfile.gettempdir()) / f"moleculeiq-{(expected or 'model')[:16]}.pth"
        if not cached.is_file():
            logger.info("Downloading model checkpoint to %s", cached)
            _download(url, cached, expected)
        else:
            _verify(cached, expected)
        return cached
