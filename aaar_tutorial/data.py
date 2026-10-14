"""TUV-x input data (cross sections, quantum yields, profiles, configuration).

The photolysis code reads these files from the directory in ``TUVX_JAX_ROOT``. They are
not part of the tuvx-jax pip package, so the first call downloads the pinned commit from
GitHub (about 5 MB) into a cache directory; later calls return immediately.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

from . import config


def ensure_photolysis_data(cache_dir: str | Path | None = None, url: str | None = None,
                           verbose: bool = True) -> Path:
    """Make the TUV-x data available, set ``TUVX_JAX_ROOT`` to it and return its path."""
    env_root = os.environ.get(config.TUVX_ROOT_ENV)
    if env_root:
        if is_complete(Path(env_root)):
            return Path(env_root)
        if verbose:
            print(f"{config.TUVX_ROOT_ENV}={env_root} lacks the TUV-x data; using the cache instead.")
    root = Path(cache_dir) if cache_dir is not None else config.TUVX_CACHE_DIR
    if not is_complete(root):
        source = url or config.TUVX_TARBALL_URL
        if verbose:
            print(f"Downloading TUV-x photolysis data (tuvx-jax {config.TUVX_JAX_COMMIT[:7]}) ...")
        t0 = time.perf_counter()
        _install(source, root)
        if verbose:
            print(f"TUV-x data ready in {root} ({time.perf_counter() - t0:.1f} s)")
    os.environ[config.TUVX_ROOT_ENV] = str(root)
    return root


def is_complete(root: Path) -> bool:
    """True if ``root`` holds every TUV-x file and folder the model needs."""
    return all((root / rel).exists() for rel in config.TUVX_REQUIRED_PATHS)


def _install(url: str, root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{root.name}.tmp-", dir=root.parent) as tmp:
        tarball = Path(tmp) / "tuvx.tar.gz"
        _download(url, tarball)
        staged = Path(tmp) / "root"
        _extract(tarball, staged)
        if not is_complete(staged):
            raise RuntimeError(f"the archive from {url} lacks {config.TUVX_REQUIRED_PATHS}")
        if root.exists():                 # incomplete leftover, or another process just won
            if is_complete(root):
                return
            shutil.rmtree(root)
        os.replace(staged, root)          # atomic: root is either absent or complete


def _download(url: str, dest: Path) -> None:
    """Download with a per-attempt deadline and a bounded number of retries."""
    attempts = config.DOWNLOAD_MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            _fetch(url, dest)
            return
        except (OSError, ValueError) as err:   # URLError, HTTPError and timeouts are OSErrors
            if attempt == attempts:
                raise RuntimeError(f"could not download {url} after {attempts} attempts: {err}") from err
            wait = config.DOWNLOAD_RETRY_WAIT_S[min(attempt, len(config.DOWNLOAD_RETRY_WAIT_S)) - 1]
            print(f"Download attempt {attempt} failed ({err}); retrying in {wait:.0f} s")
            time.sleep(wait)


def _fetch(url: str, dest: Path) -> None:
    deadline = time.monotonic() + config.DOWNLOAD_DEADLINE_S
    size = 0
    with urllib.request.urlopen(url, timeout=config.DOWNLOAD_TIMEOUT_S) as response, \
            open(dest, "wb") as out:
        while chunk := response.read(config.DOWNLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > config.TUVX_TARBALL_MAX_BYTES:
                raise ValueError(f"download exceeds {config.TUVX_TARBALL_MAX_BYTES} bytes")
            if time.monotonic() > deadline:
                raise TimeoutError(f"download took longer than {config.DOWNLOAD_DEADLINE_S:.0f} s")
            out.write(chunk)
    if size < config.TUVX_TARBALL_MIN_BYTES:
        raise ValueError(f"download is only {size} bytes (expected > {config.TUVX_TARBALL_MIN_BYTES})")


def _extract(tarball: Path, dest: Path) -> None:
    """Extract the kept top-level folders (data/, examples/) without the archive's root folder."""
    dest.mkdir()
    with tarfile.open(tarball, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or parts[1] not in config.TUVX_KEEP_DIRS:
                continue
            if not (member.isfile() or member.isdir()):
                continue
            member.name = PurePosixPath(*parts[1:]).as_posix()
            members.append(member)
        tar.extractall(dest, members=members, filter="data")
