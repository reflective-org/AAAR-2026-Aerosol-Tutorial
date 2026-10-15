"""Notebook checks: shared install cell, Colab refs, cleared outputs, pinned links; and execution.

Run the notebooks themselves with: pytest -m notebooks (about 5 minutes on a laptop).
"""
import json
import re
import time
from pathlib import Path

import pytest

from aaar_tutorial import config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = [NOTEBOOK_DIR / name for name in config.NOTEBOOKS]


def _cells(path):
    return json.loads(path.read_text())["cells"]


def _source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _install_cell(path):
    cells = [_source(c) for c in _cells(path) if c["cell_type"] == "code" and "%pip install" in _source(c)]
    assert len(cells) == 1, f"{path.name}: expected one install cell, found {len(cells)}"
    return cells[0]


def test_every_notebook_is_listed():
    assert sorted(p.name for p in NOTEBOOK_DIR.glob("*.ipynb")) == sorted(config.NOTEBOOKS)


def test_install_cells_are_identical():
    cells = {p.name: _install_cell(p) for p in NOTEBOOKS}
    assert len(set(cells.values())) == 1, "the install cells differ between notebooks"


def test_install_cell_uses_config_ref():
    cell = _install_cell(NOTEBOOKS[0])
    assert f'"{config.TUTORIAL_REQUIREMENT}"' in cell
    assert '"google.colab" in sys.modules' in cell or '"google.colab" not in sys.modules' in cell


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_colab_badge_matches_config_ref(path):
    first = _source(_cells(path)[0])
    assert f"{config.COLAB_URL}/{path.name}" in first


def test_tutorial_ref_opens_in_colab():
    # Colab splits github/<repo>/blob/<ref>/<path> at the first "/": "feat/x" opens as ref "feat".
    assert "/" not in config.TUTORIAL_REF


def test_readme_badges_match_config_ref():
    readme = (ROOT / "README.md").read_text()
    for name in config.NOTEBOOKS:
        assert f"{config.COLAB_URL}/{name}" in readme


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_are_cleared(path):
    for cell in _cells(path):
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == [] and cell.get("execution_count") is None, path.name


def test_model_links_use_the_pinned_commit():
    pattern = re.compile(r"TOMAS-JAX/(?:blob|tree)/([0-9a-f]+)/")
    for path in NOTEBOOKS + [ROOT / "README.md", ROOT / "docs" / "INSTRUCTORS.md"]:
        for commit in pattern.findall(path.read_text()):
            assert commit == config.TOMAS_JAX_COMMIT, f"{path.name} links TOMAS-JAX at {commit}"


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path):
    """Execute the notebook top to bottom in a fresh kernel, with per-cell and total time limits."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=config.NOTEBOOK_CELL_TIMEOUT_S, kernel_name="python3",
                            allow_errors=False, resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    deadline = time.monotonic() + config.NOTEBOOK_TIMEOUT_S
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(f"{path.name} took longer than {config.NOTEBOOK_TIMEOUT_S} s "
                            f"(stopped before cell {index})")
            client.timeout = int(min(config.NOTEBOOK_CELL_TIMEOUT_S, remaining)) + 1
            client.execute_cell(cell, index)
