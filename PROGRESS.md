# Progress

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
