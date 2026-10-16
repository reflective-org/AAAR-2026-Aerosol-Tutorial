# Notes for instructors

How to use the notebooks in the AAAR 2026 tutorial, how long they take, what to warn students
about, and how to update the model version. The web app has its own instructor guide (in the
tomas-api repository, `docs/aaar/INSTRUCTOR_GUIDE.md`). [model_flowchart.pdf](model_flowchart.pdf) shows how the
model works in two pages; it is a good slide to open the session with.

## Before the session

1. The repository is public and the badges and install cells point at `main`. Before the
   workshop, tag the version you tested and point them at the tag (see "Updating pins"), so a
   later push cannot change what students run. Then open every notebook from its badge in a fresh
   Colab runtime, *Runtime › Run all*, and check that it finishes. The ref must not contain a
   `/` (Colab cannot open `feat/...` branches); `scripts/set_pins.py` refuses such refs.
2. Tell students to open the notebook a few minutes early and run the first two cells: the install
   takes 1–2 minutes on Colab, and they may have to restart the session once.
3. Each Colab notebook is its own runtime and installs separately. Students who move to the next
   notebook start a new install, so let them start it while you introduce the level.

## Session plan (suggestion)

The notebooks follow the web app's three levels. A good rhythm per level: 10 minutes with the
web app's sliders to build intuition, then 25–35 minutes in the notebook for the "why" and for
changing the code.

| Block | Time | Content |
|---|---|---|
| Intro | 15 min | The box model in one picture: gas chemistry, photolysis, aerosol, coupling. Start the Level 1 install. |
| Level 1 | 40 min | c\*, partitioning, H₂SO₄ vs organic, coagulation, nucleation. Notebook sections 3–7; nucleation (8) if time allows. The partitioning slider in section 1 (no model run) makes a good 3-minute live demo. |
| Level 2 | 50 min | Explicit chemistry, oxidant shares, c\* from composition, top contributors. Discuss the c\* placeholder openly (below). Experiments 6a and 6c are the most instructive. |
| Break | 10 min | |
| Level 3 | 50 min | Sunlight, day–night radicals, N₂O₅ uptake and the γ sweep, holding NO₂ constant. |
| Wrap-up | 15 min | Questions; point to notebook 4 and ASSUMPTIONS.md. |

Notebook 4 ("Inside the model") is for students who want to see the numerics (photolysis tables,
stiff ODEs, the coupling loop). It works well as homework or as a parallel track for advanced
students.

## Run times

Measured on a laptop (Apple Silicon, CPU only), running every cell once in a fresh kernel after
installation, with the TUV-x data already downloaded. Colab is about twice as slow.

| Notebook | Model runs | Whole notebook | One run |
|---|---|---|---|
| 01 Volatility | 14 Level 1 runs | 25 s | about 1 s |
| 02 Limonene SOA | 12 Level 2 runs | 1.5 min | about 6 s |
| 03 Day & Night | 7 Level 3 runs | 2 min | about 12 s |
| 04 Inside the model | 1 hand-written Level 3 loop | 30 s | about 17 s |

The first run of each level in a session also compiles the model with JAX (about 2 s for
Level 1; for Levels 2 and 3 compiling plus building the TUV-x table adds about 5 s on this
laptop and about twice that on Colab). On Colab the first install takes 1–2 minutes.
`USE_PRECOMPUTED = True` in notebooks 1–3 loads a stored result of the preset instead of running
it.

All these times are for the default 40 size bins. The *Size resolution* cell in each notebook's
Setup (`N_BINS`: 40, 80, 160 or 320) draws finer size distributions (results change by under
1 %), but every run takes longer: Level 1 about 1, 4, 14 and 61 s, Level 3 about 12, 19, 39 and
136 s.

## Known caveats (say these out loud)

All model simplifications are in TOMAS-JAX
[ASSUMPTIONS.md](https://github.com/reflective-org/TOMAS-JAX/blob/2faa391bcf132fb6796dc71bb637974f9282a459/ASSUMPTIONS.md).
The ones students will run into:

- **c\* is a placeholder (A8).** Levels 2 and 3 estimate each product's c\* from its elemental
  composition (Li et al., 2016) until Shantanu's per-species table arrives. With it, about 98 % of
  the Level 2 SOA contains nitrogen (organic nitrates and PANs), the main non-nitrogen products
  (limonaldehyde, limononic acid) never condense, and NOx = 0 gives almost no SOA. The SOA then
  saturates near 16 µg m⁻³ when limonene goes 5 → 20 ppb because NOx runs out, so the yield falls.
  **Do not teach yield vs organic loading or low-NOx SOA with Level 2.** At NOx = 10 ppb the SOA
  (about 60 µg m⁻³) exceeds the mass of limonene reacted; the notebook calls this a red flag.
- **Level 2 light is constant (A10):** TUV-x at a solar zenith angle of 30°, scaled to
  J(NO₂) = 5×10⁻³ s⁻¹.
- **Level 3 holds NO₂ at 1 ppb (A14)** as a stand-in for NOx emissions: it is reset to 1 ppb after
  every 15-minute interval and drifts within it (0.8–0.9 ppb in daylight). That is an unlimited
  nitrogen source. With 10 ppb limonene the SOA reaches about 70 µg m⁻³, more than the limonene
  mass, because the hold keeps feeding PANs and organic nitrates. Notebook 3 says so.
- **Level 3 daytime OH is high:** 1.7×10⁷ molec cm⁻³ on the first afternoon and 2.3×10⁷ at noon on
  day 2, several times typical ambient peaks (NO₂ hold, little VOC left, O₃ rising to about 80 ppb
  without deposition).
- **Particle nitrate only comes from N₂O₅ uptake (A1, A2, A5).** γ-based uptake replaces the
  chamber's fixed N₂O₅ sink `<34>`; `<33> HNO₃ → NA` stays a gas-phase sink. There is no
  ammonia and no NH₄NO₃, so the HNO₃ the run makes (3.8 ppb by the end) stays in the gas. Nitrate appearing
  only at night is partly built into the model.
- **SOA is too hygroscopic (A9):** organics take up water like ammonium bisulfate, so wet surface,
  condensation sink and k_het are overestimated for organic-rich particles. Organics and nitrate
  also count as sulfate for density, while the reported diameters assume 1400 kg m⁻³.
- **Nucleation is H₂SO₄–H₂O only (A13).** At the Level 1 temperature (298 K) it produces no new
  particles even at 10⁷ molec cm⁻³ s⁻¹ of H₂SO₄; notebook 1 compares 298 K with 265 K. With
  10⁶ molec cm⁻³ s⁻¹, N peaks at about 7 400 cm⁻³ at 270 K and 30 000 cm⁻³ at 265 K (7 600 and 34 000 in the web app, which runs an older model). The web
  app's Level 1 prompt and tooltip say the same (fixed 2026-10-14).
- **Coupling interval (A12):** rate constants, J and the RO₂ sum are frozen for each interval
  (300 s in Levels 1–2, 900 s in Level 3). ASSUMPTIONS.md quotes ≤ 3 % between 900 s and 600 s;
  with the pinned model (40 bins) 600 s gives +3.5 % nitrate and +2.4 % SOA, and 300 s gives
  +6.6 % nitrate and +4.7 % SOA, against 900 s (not yet in ASSUMPTIONS.md).
- **Photolysis:** TUV-x gives 10–35 % more J than the MCM formula with the sun high, and 2–4× more
  J(O¹D) at solar zenith angles of 75–80° (notebook 4, Section 1).

## The web app's instructor guide

The tomas-api guide's numbers were re-measured on 2026-10-14 with tomas-jax 8753736, which the web
app still runs. The notebooks pin a newer commit (`2faa391`) that moves growing particles between
size bins with a PPM remap instead of the older MNFIX split. Both use the same 40 size bins (1.7 nm
to 17.5 µm). OH agrees: Level 3 OH ≈ 1.7×10⁷ (day 1) / 2.3×10⁷ cm⁻³ (day 2). SOA
agrees within 1 %. Particle size and nitrate differ, because the older split spreads the size
distribution: at the end of Level 1 the geometric mean diameter is 86 nm in the web app and 111 nm
in the notebooks; at the end of Level 3 (γ = 0.02) the web app
has nitrate 4.96 µg m⁻³ and a geometric mean diameter of 162 nm, the notebooks 5.16 µg m⁻³ and
185 nm. The guide counts the 725 species of the MCM file; the notebooks
say 726 because the model adds the `NA_HET` counter.

## Updating pins

Every version lives in one place per file, and the tests check that they agree.

- **This repository's ref** (Colab badges, install cells, README links): before the workshop,
  tag a commit and point everything at the tag:

  ```bash
  git tag v2026.10 && git push origin v2026.10
  python scripts/set_pins.py --tutorial-ref v2026.10
  pytest                                 # checks badges, install cells and README agree
  ```

  Commit and push the changed files (the tag itself must then be moved to that commit, or tag
  after committing the ref change).

- **The model** (TOMAS-JAX, and the tuvx-jax commit its `[photolysis]` extra pins):

  ```bash
  python scripts/set_pins.py --tomas-jax <40-character commit> [--tuvx-jax <commit>]
  uv sync --extra dev                    # or: pip install -e ".[dev]"
  python scripts/make_examples.py        # regenerates aaar_tutorial/examples/ (about 1 min)
  pytest && pytest -m notebooks          # unit tests, then every notebook (about 4 min)
  ```

  Also raise the helpers' version (below), update `VBS_JAX_COMMIT` and `TESTED_VERSIONS` in
  `aaar_tutorial/config.py` if the model changed them, and reread the notebook text: it quotes numbers from the current model
  (e.g. SOA amounts, OH peaks, nitrate) that a model change can shift. A Colab runtime that is
  already open keeps its old model (pip does not reinstall a git dependency whose version number
  is unchanged); `at.setup()` then prints a note, and *Runtime › Disconnect and delete runtime*
  gives a fresh install. Tell students to start a fresh runtime after a model change.

- **The helpers' version** (`version` in `pyproject.toml`, `__version__` in
  `aaar_tutorial/__init__.py` and `HELPERS_VERSION` in every install cell; the tests check that
  they agree): raise it with every change to `aaar_tutorial` that students need (new functions,
  fixes, changed defaults, new pins). pip keeps an installed package whose version has not
  changed, so without a raise, open Colab runtimes keep the old helpers silently. With it, the
  install cell replaces older helpers by itself, and a session that already imported them is
  asked to restart.

- **JAX and the other numerical packages** are pinned in `pyproject.toml` and `requirements.txt`
  (keep both the same; `tests/test_config.py` checks).

## Checking everything

```bash
uv run pytest                          # unit tests and notebook checks, about 10 s
uv run pytest -m notebooks             # every notebook top to bottom, about 4 min on a laptop
```

GitHub Actions (`.github/workflows/notebooks.yml`) runs both on every push.
