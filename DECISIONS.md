# Decisions

Why the tutorial repository looks the way it does. Newest last.

## Smooth growth (PPM) and 40 bins again, with a choice (2026-10-16)

The jagged size distributions at 80 bins (Ali's screenshots) came from SOA growth, not from
coagulation or H₂SO₄ condensation: SOA mass was added to each bin in place and MNFIX split bins
whose mean left them, which got worse with more bins (3 / 12 / 29 local extrema at 40 / 80 / 160
bins). TOMAS-JAX now moves SOA and nitrate between bins with a PPM remap (1f519e9, tag
aaar-stable-2026-10-16; independent verification of conservation, smoothness, PyCHAM sizes,
runtime). With it every preset is one smooth mode at every resolution and 40 bins is within
0.6 % of 160–320 bins (Level 1 mean diameter 110.8 / 111.1 / 111.1 nm at 40 / 80 / 320; Level 3
nitrate 5.160 / 5.152 / 5.149 µg m⁻³), so the default is back to the presets' 40 (Ali's choice):
fastest (Level 1 about 1 s per run) and the web app's grid. Users choose 40/80/160/320 with
`at.use_bins` (a *Size resolution* cell in every notebook; a bins dropdown in notebook 1's run
panel). The web app still runs the previous model until tomas-api's pin is updated, so its numbers
differ from the notebooks' (INSTRUCTORS.md).

Plots: the banana plot uses ColorBrewer YlGnBu (pale yellow → green → blue → navy, lightness
falling monotonically), with cells below the colour floor drawn white so empty and near-empty
cells blend into the background, and Linear/Log colour buttons. Ali chose it from four rendered
candidates (YlGnBu, YlOrBr, reversed inferno, a Reflective-brand ramp) after a one-hue blue ramp
saturated into dark navy over most of the mode. Line plots (size distributions, time series,
totals, gases, radicals) get Linear/Log buttons; stacked composition, volatility bars and the
partitioning explainer stay linear (log distorts stacked areas and percentages).

## 80 size bins in the notebooks (2026-10-15; superseded by the entry above)

The PyCHAM comparison showed that the 40-bin grid spreads growing modes numerically (mean
diameter too small by 15–25 % after a few hours of growth), while mass is right. Measured on the
presets (repeat runs, laptop): 40 / 80 / 160 bins take 1.7 / 6.2 / 19.9 s (Level 1), 6.1 / 8.3 /
15.6 s (Level 2) and 13.5 / 22.8 / 52.3 s (Level 3); the final mean diameter is 86 / 98 / 107 nm,
158 / 183 / 198 nm and 162 / 176 / 185 nm; SOA does not change and nitrate rises 2–4 %. 160 bins
would make the Level 1 run panel take about 40 s per click on Colab, so the notebooks use 80
(`config.NBINS`, applied to runs started from a preset name); `numerics={"nbins": ...}` overrides
it. The presets, the web app and its instructor guide keep 40, so their numbers differ slightly
from the notebooks'. Helpers version 0.3.0.

## One branch: `main` (2026-10-15)

Ali's call: everything is on `main`, so work happens there and `feat/aaar-tutorial` was deleted
(it was merged first). A push to `main` reaches students at once (the badges and install cells
point at it), so the unit tests and the notebook run come before every push. The other repositories
(TOMAS-JAX, tuvx-jax, tomas-api, tomas-fe) keep their `feat/aaar-tutorial` branches.

## Notebook design: images and blockquotes, because Colab strips HTML styles (2026-10-15)

The first attempt styled the markdown cells with inline HTML (`<div style=...>` cards, pills).
JupyterLab drew it well, but Colab removes `style` attributes from text cells, so on Colab, where
most students work, it fell apart into plain text and a full-size logo. The design now uses only
what every renderer (Colab, JupyterLab, VS Code, GitHub; light and dark themes) draws the same:

- **Header banner image** per notebook (`assets/banner-level1.png`): Reflective brand gradient
  (Blue #466F8D → #939280 → Gold #A47F52), Gambarino title, Open Sans text, level progress and
  "Supported by" with the white Reflective logo. Drawn by `scripts/make_banners.py` from the
  brand asset pack; the alt text carries the title.
- **Callouts are markdown blockquotes with a coloured uppercase label** (`<font color>`, which
  Colab keeps). Colour means something: gold = do something (TRY THIS, YOUR TURN, EXPLORE), green =
  look at this (WHAT TO NOTICE), slate = information (HOW TO USE IT, SETUP, LEARNING GOALS, KEY
  IDEA, THE BOX). These three read in light and dark themes; brand Blue and Clay Red are too dark
  on Colab's dark background. Two or three key sentences are highlighted with `<mark>` (Colab draws
  it yellow in light mode, amber in dark). No emojis (Ali: they look machine-made). Tested in Colab
  and JupyterLab, light and dark: coloured table cells (`bgcolor`) survive too, but Colab does not
  wrap table text and JupyterLab centres small tables, so boxes are not built from tables.
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
found". `main` is now the published version that the badges and install cells use. Before the
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
