# Progress

## 2026-10-15: Level 1 notebook design, Reflective logo

- Notebook 1 restyled with the Reflective palette (see DECISIONS.md): header card with
  "Supported by Reflective", callouts, collapsible run times and references, Colab form titles
  and fields on the experiment cells. Text and code unchanged apart from labels and the
  `#@title`/`#@param` annotations.
- `assets/reflective-logo.png` (official black logo, cropped); `config.LOGO_URL` links it at
  the commit that added it; a test keeps the notebooks on that URL.
- Checked renders: JupyterLab light and dark, Colab light and dark (headless Chrome).
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
