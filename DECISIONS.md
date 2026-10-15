# Decisions

Why the tutorial repository looks the way it does. Newest last.

## Notebook design: images and blockquotes, because Colab strips HTML styles (2026-10-15)

The first attempt styled the markdown cells with inline HTML (`<div style=...>` cards, pills).
JupyterLab drew it well, but Colab removes `style` attributes from text cells, so on Colab, where
most students work, it fell apart into plain text and a full-size logo. The design now uses only
what every renderer (Colab, JupyterLab, VS Code, GitHub; light and dark themes) draws the same:

- **Header banner image** per notebook (`assets/banner-level1.png`): Reflective brand gradient
  (Blue #466F8D → #939280 → Gold #A47F52), Gambarino title, Open Sans text, level progress and
  "Supported by" with the white Reflective logo. Drawn by `scripts/make_banners.py` from the
  brand asset pack; the alt text carries the title.
- **Callouts are markdown blockquotes** with an icon label: 🧭 how to use, 🎯 goals, 🔑 key
  idea, 🧪 Try this, 👀 what to notice, 💬 questions, ⚙️ setup, 📦 the box.
- **Collapsible extras** (`<details>`): run times, references.
- **Colab forms:** experiment cells start with `#@title` and mark their input with `#@param`,
  so Colab shows a titled section and an input box; elsewhere these are plain comments.
- **Footer badge** (`assets/supported-by-reflective.png`): white pill with the black logo,
  readable on light and dark backgrounds.

Images are linked at a fixed commit (`config.ASSET_COMMIT`) so the links work on every branch and
never change. Plots keep the web app's palette so students see the same colours in both places.

## Publish from `main`, public, MIT (2026-10-15)

Colab opens `github/<owner>/<repo>/blob/<ref>/<path>` by splitting at the first `/`, so the badges
for the branch `feat/aaar-tutorial` asked GitHub for ref `feat` and failed with "Notebook not
found". `main` is now the published version that the badges and install cells use; work continues
on `feat/aaar-tutorial` and is fast-forwarded into `main` after the tests pass. Before the
workshop the ref becomes a tag. `scripts/set_pins.py` refuses refs containing `/` and
`tests/test_notebooks.py` checks `config.TUTORIAL_REF`. The repository is public (Colab and the
install cell's `pip install git+https://...` need anonymous access) and MIT-licensed, the simplest
licence for teaching material that students copy and change.

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
`config.TUTORIAL_REF` (now `main`). Before the workshop it should become
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
