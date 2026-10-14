# Decisions

Why the tutorial repository looks the way it does. Newest last.

## A separate repository for the notebooks (2026-10-14)

The notebooks live here, not in TOMAS-JAX:

- Students open them from a Colab badge, which needs a small public repository whose
  installable package is just the helpers. TOMAS-JAX carries Fortran references, benchmarks and
  development notes that students do not need.
- The tutorial pins one model commit and can change its text and plots without touching the
  model. The model repositories stay as they are (no tutorial-only edits in them).
- The instructor and the web-app team can review the teaching material on its own.

The model repositories (TOMAS-JAX, tuvx-jax, vbs-jax) are already public, so no git submodules
are needed: pip installs them from GitHub by commit.

## Git commit pins instead of PyPI (2026-10-14)

TOMAS-JAX, tuvx-jax and vbs-jax are not on PyPI. `pyproject.toml` installs
`tomas-jax[photolysis] @ git+https://github.com/reflective-org/TOMAS-JAX@<commit>`, which pins
tuvx-jax and vbs-jax in turn. A full commit hash is immutable, so every student gets the code
the notebooks were tested with, also on Colab where pip installs from git without extra setup.
JAX, diffrax, equinox, lineax and optimistix are pinned to the tested versions because their
numerics change between releases.

The same applies to this repository's own ref: the Colab badges and the install cell use
`config.TUTORIAL_REF` (now the branch `feat/aaar-tutorial`). Before the workshop it should become
a tag, so the notebooks and the helpers they install cannot drift apart. `scripts/set_pins.py`
changes it in every file at once and the tests check that all places agree.

## Download the TUV-x data instead of shipping it (2026-10-14)

TUV-x needs cross sections, quantum yields, profiles and a configuration file that are not part
of the tuvx-jax pip package. `aaar_tutorial.ensure_photolysis_data()` downloads the pinned commit
from GitHub's archive server once (about 5 MB), keeps only `data/` and `examples/`, and points
`TUVX_JAX_ROOT` at the cache folder. The download has a per-read timeout, an overall deadline,
three attempts and size checks, and it extracts to a temporary folder that is moved into place in
one step, so a broken download never leaves a half-filled cache. Copying the data into this
repository would duplicate it and could drift from the pinned tuvx-jax commit.

## Plotly for every figure (2026-10-14)

Plotly figures work in Colab, JupyterLab and VS Code without extra setup, and students can hover
for values, zoom and hide lines. Size distributions and the volatility plot get a time slider,
like the web app. GitHub shows notebooks without Plotly output, which is acceptable: students run
the notebooks. Matplotlib would be static; Bokeh and Altair need more setup on Colab.

## Notebooks are saved without outputs (2026-10-14)

Saved outputs would go stale when the model or the text changes, would make every diff huge
(Plotly figures embed their data) and would hide whether a notebook still runs. Students run the
cells themselves; the stored preset results (`aaar_tutorial/examples/`) give instant plots where
waiting would not teach anything. `tests/test_notebooks.py` fails if a notebook has outputs.

## Plumbing in the helpers, science in the notebooks (2026-10-14)

Code that is only plumbing (progress bars, plot styling, site and time-zone handling, the TUV-x
download, writing a c\* table) lives in `aaar_tutorial`, so the notebooks stay short. Code that
*is* the lesson (the Pankow equation, the k_het estimate, the nitrate ceiling, the nitrogen
budget, the coupling loop in notebook 4) stays in the notebook where students can read and change
it.

## Custom c\* tables through the helpers (2026-10-14)

The pinned tomas-jax reads a c\* table only from its own `mechanisms/` folder. `at.run(...,
cstar_ugm3={...})` writes the table into the tutorial cache and points the model's folder there
for the duration of that run (`unittest.mock.patch`). This keeps the trick out of the notebooks
until tomas-jax accepts a table path or a dict.

## Notebook tests run through pytest and nbclient (2026-10-14)

`pytest -m notebooks` runs every notebook in a fresh kernel with a limit per cell and per
notebook. nbmake only limits each cell, and GitHub Actions' job limit is the last resort. Fast
checks (identical install cells, refs that match `config.TUTORIAL_REF`, cleared outputs, pinned
model links) run with the unit tests.
