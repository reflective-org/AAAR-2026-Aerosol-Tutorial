"""Pins agree between config.py, pyproject.toml and requirements.txt."""
import tomllib
from pathlib import Path

from aaar_tutorial import config

ROOT = Path(__file__).resolve().parents[1]


def _requirements():
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_requirements_match_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert _requirements() == project["dependencies"]


def test_model_pin_matches_config():
    assert config.TOMAS_JAX_REQUIREMENT in _requirements()
    assert config.TUVX_JAX_COMMIT in config.TUVX_TARBALL_URL


def test_tested_versions_are_pinned():
    reqs = _requirements()
    for name in ("jax", "jaxlib", "diffrax", "equinox", "lineax", "optimistix"):
        assert f"{name}=={config.TESTED_VERSIONS[name]}" in reqs
