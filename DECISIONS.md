# Decisions

Why the tutorial repository looks the way it does. Newest first.

## Level 3 coupling interval 300 s, a setting of the numerical solver (2026-10-17)

Ali, on the flowchart in Figma (comments 4 and 5): "15mins seems to be a bit long in my opinion, it
really depends how fast the chemistry is and how much emissions exist. I think that should be
defined in the numerics/solver." and "change "numerics" to "numerical solver" or something that is
clear." Asked for the Level 3 interval before the tutorial, Ali chose "300 s now, automatic later".
TOMAS-JAX 6900eb3 (tag aaar-stable-2026-10-17d) changes the Level 3 preset from 900 s to 300 s;
the tutorial moves its pin and its text:

- Why shorter is better: within one interval the rate constants, J and the RO₂ sum are frozen and
  the aerosol gets the H₂SO₄ and nitrate made in it as constant sources. That splitting is a
  first-order method, so a result X behaves as X(Δt_c) ≈ X₀ + a·Δt_c, with X₀ the result with
  continuous coupling. Against a run with 30 s intervals (TOMAS-JAX `docs/aaar/coupling.md`), the
  Level 3 nitrate at 24 h is 6.4 % low at 900 s, 2.0 % at 300 s and 0.2 % at 60 s, and the SOA
  38 %, 10 % and 1.8 % high.
- Why not shorter: a 24 h Level 3 run takes about 25 s at 300 s and about 90 s at 60 s on a laptop
  (13 s at 900 s), and notebook 3 makes 7 runs in a 50-minute session.
- Words: where the text explains the interval it calls it a setting of the numerical solver and
  gives the equation above; `numerics={...}` stays in code because that is what students type.
- Planned after 2026-10-27, not part of this change: the numerical solver chooses the interval
  itself, from how fast the chemistry changes and how much is emitted. The notebooks, README,
  INSTRUCTORS and the flowchart say so where they discuss the interval.
- Every Level 3 number in the text was measured again at 300 s; the scripts reproduced the old
  numbers at 900 s first (PROGRESS lists both).
- Helpers 0.4.9: the pin changed, and pip keeps installed helpers whose version is unchanged.

## TUV-x data that opens in any Windows folder: tuvx-jax 655668c (2026-10-17)

On Windows, the netCDF library that tuvx-jax used to open its data files cannot open a file
whose folder path has non-ASCII characters. The laptop app's Windows test found it with the
cache folder "cache-José-データ". The tutorial keeps the TUV-x data in the user's home folder
(`~/.cache/aaar_tutorial`), so a student whose Windows user name has an accent would hit it in
`at.setup()`. The fix is in tuvx-jax, the code that opens the files: from 655668c it reads each
file into memory and opens it from there, and all 172 data files give the same arrays as before.
The notebooks, the web app and the laptop app all get it from that one place; the tutorial only
moves its pins:

- TOMAS-JAX 44b6db2 (tag aaar-stable-2026-10-17c), whose `[photolysis]` extra pins tuvx-jax
  655668c. Its model code is that of 0e23da0.
- `TUVX_JAX_COMMIT` 655668c, so the data download and the installed tuvx-jax come from the same
  commit.
- Helpers 0.4.8: the pins changed, and pip keeps installed helpers whose version is unchanged.

Colab runs Linux and was never affected. On their own computers students only need the usual
update (`git pull`, then `uv sync`), so the README gets no new troubleshooting item. Moving the cache to an ASCII-only
folder in the helpers would have fixed only the notebooks, and it would need a folder outside the
home folder that every student can write to.

## Nucleation runs take the model's short time steps, in notebook 4 too (2026-10-17)

Ali: "for nucleation, 15 mins interval is way too long, we need to have a shorter timesteps for
nucleation cases ... 1-2 mins max". TOMAS-JAX 0e23da0 (tag aaar-stable-2026-10-17b) does this in
the model: with nucleation on, the coupling interval is cut into equal parts of at most 60 s and
the aerosol micro-step is at most 5 s (ASSUMPTIONS A13). Each micro-step adds its H₂SO₄ at once,
C ← C + P·Δt_m, and the binary nucleation rate goes as J = k(T)·Cⁿ with n ≈ 4 (Dunne et al.,
2016, the scheme's binary neutral term), so 60 s micro-steps overshoot the burst (peak N +30 % in
Level 2 and +78 % in Level 3 at 265 K); 60 s / 5 s is within 4 % in every 40-bin case. The
tutorial follows the model and adds no steps of its own:

- `at.run` passes the scenario on unchanged, so the caps and the model's warning line reach the
  student as they are.
- Notebook 4's hand-written loop asks `driver.time_steps(s, cfg)` for the coupling interval and
  the micro-step instead of reading `s.numerics.coupling_dt_s`, and passes the micro-step to
  `aerosol_outer_step` as `driver.run` does. Without nucleation the two values are the old ones
  (900 s, 60 s), so the default loop is unchanged; with nucleation on it matches `at.run`.
- The helpers' time limits stay: `RUN_TIMEOUT_S` (900 s per run) and `NOTEBOOK_CELL_TIMEOUT_S`
  (900 s per cell) already allow the slowest 40-bin nucleation run, Level 3 over 48 h (about
  225 s on a laptop, so about 450 s on Colab). Finer grids with nucleation can exceed 900 s; the
  `TimeoutError` says how to pass a larger `timeout_s`.
- The model now checks every scenario value where it is built (RH 80 gives "did you mean 0.8?",
  `coagulation="off"` is refused). The helpers check only input names (with spelling
  suggestions), never values, so there is no second set of value checks to keep in step; tests
  make sure the model's message reaches the student word for word.

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
