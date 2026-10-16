# Progress

## 2026-10-17: nucleation time steps and input checks (helpers 0.4.7)

- Model pin TOMAS-JAX 0e23da0 (tag aaar-stable-2026-10-17b). With nucleation on, the model cuts
  each coupling interval into equal parts of at most 60 s and uses 5 s aerosol micro-steps instead
  of 60 s (A13: each micro-step adds its H₂SO₄ at once and J ∝ [H₂SO₄]ⁿ with n ≈ 4, so 60 s steps
  put the peak particle number up to 78 % high). The model also checks every scenario value where
  it is built (RH 80: "did you mean 0.8?"; process switches must be True or False).
- Runs without nucleation are unchanged: the regenerated examples agree with the old ones in every
  stored value (6 digits); only their diagnostics gain the requested and used time steps.
- Notebook 4: the hand-written loop takes the coupling interval and micro-step from
  `driver.time_steps`, so with nucleation on it uses the same 60 s / 5 s steps as `run()`. Checked
  once with nucleation on (Level 3, 24 h, 1440 intervals): loop and model agree to 6 digits in
  nitrate (3.34413 µg m⁻³), SOA, number, O₃, HNO₃ and solver steps (8664); loop 128 s, model run
  121 s. Against the preset (nucleation off), SOA −27.0 % and nitrate +6.6 %, with no new particles.
- Text: notebook 4 Section 5 explains the shorter steps with the H₂SO₄ step and J = k(T)Cⁿ, and the
  interval split Δt_c = Δt_req/m; notebook 1 Experiment 4 says why nucleation runs take about 15 s
  (1 min at 80 bins); notebooks 2 and 3 note the steps and run times (Level 3 also the SOA and
  nitrate shift); README and INSTRUCTORS give the run times and the input-check message.
- Measured (laptop, warm, 40 bins unless noted): Level 1 with nucleation 13–15 s (58–60 s at
  80 bins), Level 2 27 s, Level 3 24 h about 2 min, Level 3 at 80 bins 50 s per 6 h (so about
  3.5 min per 24 h). Level 1 at 1e6 cm⁻³ s⁻¹ H₂SO₄: peak N 7 071 cm⁻³ at 270 K and 29 420 at
  265 K (INSTRUCTORS had 7 400 and 30 000 from the 60 s steps).
- Helper tests: the model's messages for RH 80 and `coagulation="off"` reach the student word for
  word through `at.scenario_from` and `at.run`. No helper-side value checks existed, so none were
  removed.
- Time limits unchanged: `RUN_TIMEOUT_S` 900 s and `NOTEBOOK_CELL_TIMEOUT_S` 900 s cover the
  slowest 40-bin nucleation run (Level 3, 48 h, about 225 s on a laptop).
- Tests: pytest 157 passed; pytest -m notebooks 4 passed in 3 min 47 s (01 47 s, 02 66 s,
  03 90 s, 04 24 s).
- Open: `docs/model_flowchart.pdf` still shows only the steps without nucleation (5/15 min
  intervals, 60 s micro-steps) and names 5227ddd; README and INSTRUCTORS say so until it is
  re-printed.

## 2026-10-16 (evening): Shantanu's c\* table is the Level 2/3 default (helpers 0.4.5)

- Model pin TOMAS-JAX 5227ddd (tag aaar-stable-2026-10-16c): Levels 2 and 3 take each product's
  c\* from Shantanu Jathar's per-species table (Nannoolal et al. 2008 vapour pressures via
  PyCHAM 5.8.0, 298.15 K; ASSUMPTIONS A8) instead of the Li et al. (2016) composition estimate.
  Level 1 is unchanged. Stored examples regenerated (40 bins): Level 2 SOA 15.86 → 8.66 µg m⁻³,
  nitrogen-containing share of the top contributors 0.99 → 0.03 (LIMALOOH and other
  hydroperoxides lead); Level 3 SOA 7.85 → 0.06 µg m⁻³ and nitrate 5.16 → 3.14 µg m⁻³.
- Level 3 SOA is now very small (peak 0.39 µg m⁻³ at 2 h): Ali chose to keep the preset and
  explain it (daytime OH and sunlight destroy the gas-phase hydroperoxides, so the SOA
  evaporates).
- `at.run(cstar_ugm3=...)` now merges the given values into the full default table; before, it
  wrote only those rows and every other species fell back to Li et al. (2016). Tests for the
  merge and for a short Level 2 run with it.
- README and INSTRUCTORS: the c\* caveat rewritten for the table (with the c\* formula), a new
  Level 3 SOA caveat, and the Level 2/3 numbers (OH 1.2×10⁷ / 2.4×10⁷ cm⁻³, HNO₃ 3.6 ppb,
  coupling-interval sensitivity, web app numbers) re-taken from the new examples and the impact
  measurements.
- Flowchart (`docs/model_flowchart.pdf`): c\* source on page 1 is the table (Li et al. 2016 only
  for species not in it), MNFIX cites Adams and Seinfeld (2002), the reference list gains both
  papers, and the footers name 5227ddd. Re-printed, no box overflows.

## 2026-10-16 (later): number audit, Colab panels, flowchart, nitrate gate fix (helpers 0.4.3)

- Audit of every quantitative claim in the notebooks, README and INSTRUCTORS against the PPM
  model at 40 bins: 62 edits (run times, 80-bin wording, a dozen shifted numbers, a notebook 4
  loop-check bug). All four notebooks re-executed.
- Colab: the run panels (notebooks 1-3) drew no plots (Plotly inside a widget output on Colab:
  "Plotly is not defined"). `plots.show_in_widget` puts each figure in an iframe page on
  Colab only (helpers 0.4.2). Not yet confirmed in a real Colab session.
- `docs/model_flowchart.pdf`: two-page flowchart, checked statement by statement against the
  code (25 corrections before release).
- Model pin TOMAS-JAX 2faa391 (tag aaar-stable-2026-10-16b): nitrate from N2O5 uptake now goes
  to every bin with particles, not only bins above 1 cm-3. That removes the faint 40-90 nm band
  left behind after sunset in Level 3 (it grew with resolution; now it converges). Level 3
  nitrate 5.1602 -> 5.1597 at 40 bins; Levels 1-2 unchanged. Examples regenerated.

## 2026-10-16: PPM model, 40 bins with a choice, plot buttons (helpers 0.4.0)

- Model pin TOMAS-JAX 1f519e9 (PPM bin transfer for SOA and nitrate; see DECISIONS.md).
- `at.use_bins` and a *Size resolution* cell in every notebook (40/80/160/320, default 40);
  bins dropdown in notebook 1's run panel. Stored examples regenerated (40 bins, new model).
- Banana plot white → dark blue with a white floor; Linear/Log buttons on line plots and the
  banana colour scale (checked by clicking them in headless Chrome).
- `compare_size_distributions` works for runs on different size grids (it crashed before).
- References for PPM, Simmel et al. (2002) and Tzivion et al. (1987) in notebook 1.
- Mistake: commit cff7a60 was pushed with one failing test (the test runner's exit code was
  hidden by a pipe); fixed in af69f5a minutes later. Commits are now gated on pytest's exit code.
- Text of all notebooks, README and INSTRUCTORS re-checked against the new defaults (audit
  workflow); see the commit that applies its edits.

## 2026-10-15: 80 size bins, C_OA slider to 1000, labels without emojis (helpers 0.3.0)

- Runs started from a preset name use 80 size bins (`config.NBINS`; see DECISIONS.md for the
  40/80/160 measurements). Stored examples regenerated at 80 bins; the notebook text was checked
  number by number against 80-bin runs (one agent per notebook).
- Partitioning slider: C_OA from 0.1 to 1000 µg m⁻³ (wildfire smoke) in 81 steps.
- Notebook 1: emoji labels replaced by coloured uppercase labels, two key sentences highlighted,
  Colab form titles without emojis; α = 1 explained; wildfire range in the slider caption.
- Helpers version 0.3.0 so open Colab runtimes replace the 0.2.0 helpers.
- Checked against the Reflective brand guidelines: banner eyebrow now Open Sans Regular with 6 %
  tracking and subtitle Open Sans Light (headline Gambarino, Blue→Gold gradient, logo ≥ 100 px).
- Text audit at 80 bins: about 300 claims checked in the four notebooks, README and INSTRUCTORS;
  42 edits (run times, diameters, nitrate 5.0 → 5.1, nucleation peaks, the raw-API cells that
  keep the preset's 40 bins). All four notebooks run top to bottom (6.3 min on a laptop).

## 2026-10-15: helpers version 0.2.0; install cells update old helpers

Ali hit two errors with the new partitioning plot: on Colab `NameError: plots` (the cell ran
without the setup cells, e.g. after a restart), locally `AttributeError: ... no attribute
'partitioning'` (the kernel had imported the helpers before the update).

- The partitioning cell imports `plots` itself, so it runs on its own.
- Install cells (identical in all four notebooks) now check the installed helper version against
  `HELPERS_VERSION = "0.2.0"`. On Colab an older install is replaced (pip replaces a different
  version by itself); elsewhere the error says to pick the `.venv` kernel or `git pull` + `uv sync`
  and restart. If older helpers are already imported, the cell asks for a restart instead of
  failing later on a missing function.
- Verified in simulated Colab runtimes (fresh; older helpers installed; older helpers imported)
  and local ones (old, empty and current environments), plus all tests and all four notebooks.
  The Setup text no longer promises "at most one restart per runtime".
- Version 0.2.0 in `pyproject.toml` and `aaar_tutorial.__version__`; a test checks all three agree.
- `setup()` says how to get a fresh install when the installed tomas-jax is not the pinned one.

## 2026-10-15: partitioning explainer; work on `main` only

- `plots.partitioning()`: equilibrium partitioning F_p = 1/(1 + c*/C_OA) for c* = 0.01–10⁴ µg m⁻³
  with C_OA (0.1–100 µg m⁻³) on a slider and play button; no model run, so it reacts instantly
  and works in a saved notebook. One class can be outlined and named (`highlight_cstar`,
  `highlight_label`). In notebook 1 it follows the absorptive-partitioning equation (section 1),
  with a Try this (which C_OA puts the surrogate 90 % in the particles?). Test added.
- `_add_time_slider` takes a prefix, suffix, label function and start step, so it drives this
  slider too; time sliders are unchanged.
- Branches: `feat/aaar-tutorial` merged into `main` and deleted (Ali's call); see DECISIONS.md.

## 2026-10-15: Level 1 notebook design, "Supported by Reflective"

- Notebook 1 redesigned (see DECISIONS.md): header banner image with the Reflective logo,
  blockquote callouts with icons, collapsible run times and references, Colab form titles and
  input fields on the experiment cells, a "Supported by Reflective" badge at the end. Text and
  code are unchanged apart from labels and the `#@title`/`#@param` annotations.
- First attempt (inline HTML styles) dropped: Colab strips `style` attributes.
- `scripts/make_banners.py` draws the banner and badge from the Reflective brand assets;
  `config.ASSET_COMMIT` pins the image links; a test checks every image link.
- Checked in headless Chrome: Colab (logged out; all images, boxes, forms and the badge render)
  and JupyterLab, light and dark themes. Notebook 1 still runs top to bottom.
- Next: the same design for notebooks 2–4 once Ali approves Level 1.

## 2026-10-15: published on `main`, public, MIT licence

- The Colab badges failed ("Notebook not found"): Colab cannot open a ref with a `/` such as
  `feat/aaar-tutorial`. Badges, install cells and README now use `main` (`set_pins.py --tutorial-ref
  main`); `main` was fast-forwarded to the branch. `set_pins.py` refuses refs with `/`, and a test
  checks `config.TUTORIAL_REF`. Fixed `set_pins.py`'s README phrase so it matches on the next run.
- Repository made public; MIT `LICENSE` added (`pyproject.toml` `license = "MIT"`).
- Checked that the local install in the README works from a fresh clone (`uv sync --extra dev`,
  notebook 1 top to bottom in 43 s with an empty data cache).
- Next: a real Colab run of every badge (needs a Google login), then a tag before the workshop.

## 2026-10-14: notebooks complete, review findings applied

**State.** Four notebooks (Level 1 volatility, Level 2 limonene SOA, Level 3 day and night,
inside the model), the `aaar_tutorial` helper package, tests, CI and docs are on branch
`feat/aaar-tutorial`. All notebooks run top to bottom headless; all tests pass.

**Done today**

- Wrote the four notebooks against TOMAS-JAX 8753736 (tuvx-jax e5d9032, vbs-jax 4a67f16).
- Applied the student and science reviews:
  - Install and Colab: one identical install cell in all notebooks (installs only on Colab,
    otherwise points to the README). Badges and install now use the same ref
    (`config.TUTORIAL_REF`, now `feat/aaar-tutorial`). The stale-package check runs when
    `aaar_tutorial` is imported, before numpy or jax load. The setup text warns about pip's
    "dependency conflicts" lines and the one possible restart.
  - Helpers: clearer errors for misspelled inputs (with suggestions), `seed={"number_cm3": ...}`
    shorthand, `run(progress=callable)`, `run(cstar_ugm3={...})` for custom c\* tables,
    `mechanism()`, `site_overrides()`, `sunlight_over_day()`, `is_night()`, `night_mean()`,
    `ugm3_per_ppb()`, `cstar_at()`, `nitrogen_share()`, `value_at()`, new series `gmd_nm` and
    `org_gas_ugm3`, `plots.style()` and `plots.sunlight()`. Inline plumbing in the notebooks was
    replaced by these; the science calculations stay inline.
  - Plots: readable log-axis ticks, gaps instead of drops to the axis floor for zeros, units on a
    second line, taller stacked panels, N₂O₅ in ppt, one dotted initial curve per run when runs
    start differently.
  - Science text: c\* definition, Kelvin and k_het validity statements, the model-structure
    reasons for night-only nitrate, the k_het estimate's cancelling errors, A9 (SOA water uptake)
    where it matters, the high Level 3 OH, the "more limonene" artefact, 726 species, the
    298.15 K c\* correction, and smaller wording fixes.
  - Consistency: `USE_PRECOMPUTED` in notebooks 1–3, χ for the solar zenith angle, one heading
    style, renamed notebook 4 Section 5 variables so earlier sections can be rerun.
- Regenerated the stored preset results with the pinned model (values unchanged to 6 digits;
  code version now 30bc43b).
- Added README, docs/INSTRUCTORS.md, DECISIONS.md, `scripts/set_pins.py`,
  `scripts/make_examples.py`, and a GitHub Actions workflow that runs the tests and every notebook.

**Verification.** `pytest`: all unit tests pass (about 30 s). `pytest -m notebooks`: all four
notebooks pass on a laptop (01: 35 s, 02: 1.5 min, 03: 1.75 min, 04: 30 s).

**Open**

- The repository is private. Colab badges and the install cell only work once it is public.
  Before the workshop: make it public, tag a release, run `scripts/set_pins.py --tutorial-ref
  <tag>`, then open each notebook from its badge in a fresh Colab runtime and run all.
- Not yet checked on real Colab: install time, the restart behaviour, widget panels and Plotly
  figures inside ipywidgets output.
- The repository has no license yet; pick one before making it public.
- For Shantanu: the curated c\* table (A8); the web app's nucleation prompt shows nothing at
  298 K (A13); coupling-interval sensitivity for Level 3 (A12); TUV-x vs MCM J(O¹D) framing;
  two Level 2 statements (low-NOx limonene SOA in chambers; yield above 1 at NOx = 10 ppb).
- For the tomas-api/tomas-web owners: instructor guide numbers (Level 3 OH, nitrate, species
  count) and the nucleation tooltip (see docs/INSTRUCTORS.md).
- For tomas-jax: accept a c\* table path or dict, so `run(cstar_ugm3=...)` no longer needs to
  patch `driver.MECHANISM_DIR`; tag the pinned commits (they are branch heads now).
