"""ensure_photolysis_data against a local fake tarball (no network)."""
import io
import tarfile
import time

import pytest

from aaar_tutorial import config, data


def _fake_tarball(path, top="tuvx-jax-abc123", extra=b""):
    files = {f"{top}/{rel}/x.txt" if not rel.endswith(".json") else f"{top}/{rel}": b"x"
             for rel in config.TUVX_REQUIRED_PATHS}
    files[f"{top}/tuvx_photolysis/core.py"] = b"code"            # must be dropped
    files[f"{top}/data/padding.bin"] = b"\0" * config.TUVX_TARBALL_MIN_BYTES + extra
    with tarfile.open(path, "w:gz", compresslevel=0) as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo(f"{top}/data/evil")                 # must be skipped
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    return path.as_uri()


@pytest.fixture(autouse=True)
def restore_env(monkeypatch):
    """ensure_photolysis_data sets TUVX_JAX_ROOT; put the real value back after each test."""
    monkeypatch.setenv(config.TUVX_ROOT_ENV, "")
    monkeypatch.delenv(config.TUVX_ROOT_ENV)


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(config.TUVX_ROOT_ENV, raising=False)


def test_download_extract_and_reuse(tmp_path, no_env, monkeypatch):
    url = _fake_tarball(tmp_path / "fake.tar.gz")
    cache = tmp_path / "cache" / "tuvx"
    root = data.ensure_photolysis_data(cache_dir=cache, url=url, verbose=False)
    assert root == cache and data.is_complete(root)
    assert sorted(p.name for p in root.iterdir()) == ["data", "examples"]
    assert not (root / "data" / "evil").exists()
    assert list(cache.parent.iterdir()) == [cache]                # no temp leftovers
    import os
    assert os.environ[config.TUVX_ROOT_ENV] == str(cache)

    monkeypatch.delenv(config.TUVX_ROOT_ENV)
    t0 = time.perf_counter()
    data.ensure_photolysis_data(cache_dir=cache, url="file:///nonexistent", verbose=False)
    assert time.perf_counter() - t0 < 0.1                        # cached: no download


def test_valid_env_root_is_used(tmp_path, monkeypatch):
    url = _fake_tarball(tmp_path / "fake.tar.gz")
    root = data.ensure_photolysis_data(cache_dir=tmp_path / "a", url=url, verbose=False)
    monkeypatch.setenv(config.TUVX_ROOT_ENV, str(root))
    assert data.ensure_photolysis_data(cache_dir=tmp_path / "b", url="file:///nope") == root


def test_invalid_env_root_falls_back_to_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(config.TUVX_ROOT_ENV, str(tmp_path / "empty"))
    url = _fake_tarball(tmp_path / "fake.tar.gz")
    root = data.ensure_photolysis_data(cache_dir=tmp_path / "c", url=url, verbose=False)
    assert root == tmp_path / "c"


def test_too_small_download_fails_after_bounded_retries(tmp_path, no_env, monkeypatch):
    tiny = tmp_path / "tiny.tar.gz"
    tiny.write_bytes(b"not a tarball")
    monkeypatch.setattr(config, "DOWNLOAD_RETRY_WAIT_S", (0.0, 0.0))
    with pytest.raises(RuntimeError, match=f"after {config.DOWNLOAD_MAX_ATTEMPTS} attempts"):
        data.ensure_photolysis_data(cache_dir=tmp_path / "d", url=tiny.as_uri(), verbose=False)
    assert not (tmp_path / "d").exists()
