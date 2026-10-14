"""Detect packages that were upgraded after this Python session imported them (Colab).

Only uses the standard library, so ``aaar_tutorial/__init__.py`` can run the check before
it imports numpy, jax or plotly.
"""
import sys
from importlib import metadata

from . import config

RESTART_MESSAGE = ("Installed, but this Python session still has the old versions loaded. "
                   "Colab: click Runtime > Restart session, then Runtime > Run all. "
                   "Locally: restart the kernel.")


def stale_imports() -> list[str]:
    """'name loaded -> installed' for every sensitive package whose loaded version is outdated."""
    stale = []
    for module_name, dist in config.UPGRADE_SENSITIVE:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        loaded = getattr(module, "__version__", installed)
        if loaded != installed:
            stale.append(f"{module_name} {loaded} -> {installed}")
    return stale


def check_session() -> None:
    """Raise a RuntimeError that asks for a restart if a sensitive package is stale."""
    stale = stale_imports()
    if stale:
        raise RuntimeError(f"{RESTART_MESSAGE} (upgraded: {', '.join(stale)})")
