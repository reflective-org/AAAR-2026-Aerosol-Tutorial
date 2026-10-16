# Notes for instructors

How to use the notebooks in the AAAR 2026 tutorial, how long they take, what to warn students
about, and how to update the model version. The web app has its own instructor guide (in the
tomas-api repository, `docs/aaar/INSTRUCTOR_GUIDE.md`). [model_flowchart.pdf](model_flowchart.pdf) shows how the
model works in two pages; it is a good slide to open the session with. Its time steps (5 or 15 min
intervals, 60 s micro-steps) are those without nucleation; with nucleation on the model uses at
most 60 s and 5 s (see Run times).

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
| Level 2 | 50 min | Explicit chemistry, oxidant shares, c\* from a group-contribution table, top contributors (mostly hydroperoxides). Say how uncertain c\* is (below). Experiments 6a (more NOx gives less SOA) and 6c (the yield rises with limonene) are the most instructive. |
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
| 01 Volatility | 14 Level 1 runs (2 with nucleation) | 50 s | about 1 s (15 s with nucleation) |
| 02 Limonene SOA | 12 Level 2 runs | 1.5 min | about 6 s |
| 03 Day & Night | 7 Level 3 runs | 2 min | about 12 s |
| 04 Inside the model | 1 hand-written Level 3 loop | 30 s | about 17 s |

The first run of each level in a session also compiles the model with JAX (about 2 s for
Level 1; for Levels 2 and 3 compiling plus building the TUV-x table adds about 5 s on this
laptop and about twice that on Colab). On Colab the first install takes 1–2 minutes.
`USE_PRECOMPUTED = True` in notebooks 1–3 loads a stored result of the preset instead of running
it.

**Nucleation makes runs slower.** With `processes={"nucleation": True}` the model cuts the coupling
interval to at most 60 s and the aerosol micro-step from 60 s to 5 s (TOMAS-JAX A13), so a run takes
about 15 s in Level 1 (about 1 min at 80 bins), 25 s in Level 2 and 2 min in Level 3 at 40 bins,
and a 48 h Level 3 run about 4 min; Colab can be twice as slow. The reason is physical: each
micro-step adds the H₂SO₄ produced during it at once, and the binary nucleation rate grows with
about the fourth power of the H₂SO₄ concentration, J = k(T)·[H₂SO₄]ⁿ with n ≈ 4
([Dunne et al., 2016](https://doi.org/10.1126/science.aaf2649)), so 60 s micro-steps overshoot the
burst (peak particle number up to 78 % too high in Level 3); 5 s micro-steps keep it within 4 % at
40 bins. Notebook 1's Experiment 4 makes two such runs; the other notebooks run nucleation only if
a student switches it on.

All these times are for the default 40 size bins. The *Size resolution* cell in each notebook's
Setup (`N_BINS`: 40, 80, 160 or 320) draws finer size distributions (results change by under
1 %), but every run takes longer: Level 1 about 1, 4, 14 and 61 s, Level 3 about 12, 19, 39 and
136 s.

## Known caveats (say these out loud)

All model simplifications are in TOMAS-JAX
[ASSUMPTIONS.md](https://github.com/reflective-org/TOMAS-JAX/blob/0e23da0da79a5a080baa02447e47951b75f796fe/ASSUMPTIONS.md).
The ones students will run into:

- **c\* comes from Shantanu's per-species table, and it is still the most uncertain input (A8).**
  For each product of the Level 2 and 3 chemistry, the table gives the pure-component vapour
  pressure at 298.15 K from the group-contribution method of
  [Nannoolal et al. (2008)](https://doi.org/10.1016/j.fluid.2008.04.020), computed with
  [PyCHAM](https://doi.org/10.5194/gmd-14-675-2021) 5.8.0, converted to a saturation
  concentration:

  c\* = 10⁶ · M · p_sat / (R · T)

  where c\* is in µg m⁻³, M is the molar mass (g mol⁻¹), p_sat the vapour pressure (Pa),
  R = 8.314 J mol⁻¹ K⁻¹ and T = 298.15 K. Limonene, CO, CH₄ and a tracer are not in the table;
  they get the composition-only estimate of [Li et al. (2016)](https://doi.org/10.5194/acp-16-3327-2016),
  and none of them condenses. What students will see in Level 2:
  - The SOA (8.7 µg m⁻³ after 6 h) is mostly hydroperoxides, the –OOH products of RO₂ + HO₂:
    LIMALOOH alone is 3.7 µg m⁻³. Only about 5 % of it contains nitrogen; most PANs stay in
    the gas.
  - NOx = 0 gives 8.5 µg m⁻³ and NOx = 10 ppb only 1.0 µg m⁻³: with more NO the peroxy radicals
    react with NO instead of HO₂, so fewer hydroperoxides form.
  - From 5 to 20 ppb limonene the SOA goes from 3.0 to 17.8 µg m⁻³ and the yield (SOA formed per
    limonene reacted, by mass) from 0.11 to 0.16: the textbook rise of yield with organic
    loading.

  So yields and the NOx dependence can now be taught, qualitatively. Two limits: below about
  5 ppb limonene the yield collapses mostly because the chemistry changes (more OH, more RO₂ + NO),
  not because of C_OA (notebook 2, 6c); and MCM v3.3.1 has no peroxy-radical autoxidation or
  RO₂ + RO₂ dimers ([Jokinen et al., 2015](https://doi.org/10.1073/pnas.1423977112)), so "mostly
  hydroperoxides" describes this model, not necessarily real limonene SOA. Say out loud: other
  group-contribution methods can differ by a decade or more for multifunctional products, and
  raising every c\* by one decade removes about three quarters of the Level 2 SOA (notebook 2,
  Section 4); the table's 298.15 K values serve as the model's 298 K reference, which makes c\*
  1.8–3.3 % too high; there is no particle-phase chemistry. Any scenario can be rerun with the
  old composition-only estimate for every product (`cstar_table=None`): Level 2 then gives
  15.9 µg m⁻³, 98 % of it nitrogen-containing.
- **Level 3 makes very little SOA (A8).** It peaks at 0.4 µg m⁻³ about 2 h in and ends at
  0.06 µg m⁻³ after 24 h, so the SOA panels of notebook 3 and the web app are almost empty. With
  2 ppb limonene there is little organic mass for the products to dissolve into. With NO₂ held at
  1 ppb only about 7 % of the peroxy radicals react with HO₂ in the first two hours (a third in
  Level 2), so few hydroperoxides form, and the daytime OH and sunlight destroy those in the gas,
  so the particles give theirs back. Without the hold the SOA ends at 0.8 µg m⁻³. The preset is
  kept on purpose. With 10 ppb limonene the limonene and its products use up OH (first-afternoon
  peak 2.6×10⁶ instead of 1.2×10⁷ cm⁻³), and the SOA reaches 9.7 µg m⁻³; nitrate falls
  from 3.1 to 1.3 µg m⁻³ because the extra limonene and its products take up NO₃ that would
  otherwise form N₂O₅ (night-time N₂O₅ falls by three quarters). Particle nitrate in the preset
  is 3.1 µg m⁻³ (5.2 with the old c\*), because less SOA and less water on the particles leave
  less surface for N₂O₅ uptake.
- **Level 2 light is constant (A10):** TUV-x at a solar zenith angle of 30°, scaled to
  J(NO₂) = 5×10⁻³ s⁻¹.
- **Level 3 holds NO₂ at 1 ppb (A14)** as a stand-in for NOx emissions: it is reset to 1 ppb after
  every 15-minute interval and drifts within it (about 0.8–1.2 ppb in daylight, typically 0.9). That is an unlimited
  nitrogen source: the box ends the day with about four times more inorganic nitrogen than it
  started with (1.5 → 6.0 ppb). Notebook 3 compares a run without the hold.
- **Level 3 daytime OH is high:** 1.2×10⁷ molec cm⁻³ on the first afternoon and 2.4×10⁷ at noon on
  day 2, the second several times typical ambient peaks (NO₂ hold, little VOC left, O₃ rising to
  about 84 ppb without deposition).
- **Particle nitrate only comes from N₂O₅ uptake (A1, A2, A5).** γ-based uptake replaces the
  chamber's fixed N₂O₅ sink `<34>`; `<33> HNO₃ → NA` stays a gas-phase sink. There is no
  ammonia and no NH₄NO₃, so the HNO₃ the run makes (3.6 ppb by the end) stays in the gas. Nitrate appearing
  only at night is partly built into the model.
- **SOA is too hygroscopic (A9):** organics take up water like ammonium bisulfate, so wet surface,
  condensation sink and k_het are overestimated for organic-rich particles. Organics and nitrate
  also count as sulfate for density, while the reported diameters assume 1400 kg m⁻³.
- **Nucleation is H₂SO₄–H₂O only (A13).** At the Level 1 temperature (298 K) it produces no new
  particles even at 10⁷ molec cm⁻³ s⁻¹ of H₂SO₄; notebook 1 compares 298 K with 265 K. With
  10⁶ molec cm⁻³ s⁻¹, N peaks at about 7 100 cm⁻³ at 270 K and 29 000 cm⁻³ at 265 K (with the
  5 s micro-steps; the old 60 s steps gave 7 400 and 30 000). The web
  app's Level 1 prompt and tooltip say the same (fixed 2026-10-14).
- **Coupling interval (A12):** rate constants, J and the RO₂ sum are frozen for each interval
  (300 s in Levels 1–2, 900 s in Level 3; at most 60 s with nucleation on). Against 900 s, the Level 3 preset (40 bins) gives
  +2.6 % nitrate and −10.5 % SOA with 600 s, and +4.7 % nitrate and −20 % SOA with 300 s. The
  SOA is so small (0.06 µg m⁻³ after 24 h) and so much of it evaporates on the second morning
  that it reacts strongly to the interval; nitrate and HNO₃ change by a few percent. Because
  nucleation shortens the interval to 60 s, switching it on in Level 3 changes SOA (−27 %) and
  nitrate (+6.6 %) at 24 h even though the preset makes no new particles at 293 K; the 60 s values
  are the accurate ones (within 0.8 % of a 30 s run). A fair on/off comparison runs the case
  without nucleation at `numerics={"coupling_dt_s": 60}` too.
- **Photolysis:** TUV-x gives 10–35 % more J than the MCM formula with the sun high, and 2–4× more
  J(O¹D) at solar zenith angles of 75–80° (notebook 4, Section 1).

## The web app's instructor guide

Since 2026-10-17 the web app (tomas-api and tomas-web, branch `feat/aaar-tutorial`) runs the same
TOMAS-JAX commit as the notebooks (`0e23da0`, with Shantanu's c\* table and the shorter nucleation
time steps) on the same 40 size bins (1.7 nm to 17.5 µm). The tomas-api guide's numbers were
measured through the API's own code path, most of them on 2026-10-16 with `5227ddd` (which gives
the same results for runs without nucleation) and every number that involves nucleation again on
2026-10-17 with `0e23da0`, so the two agree. Shared numbers: Level 1
geometric mean diameter 111 nm, Level 2 SOA 8.7 µg m⁻³, Level 3 nitrate 3.1 µg m⁻³, Level 3
OH ≈ 1.2×10⁷ (day 1) / 2.4×10⁷ cm⁻³ (day 2).
Before the table the web app ran 2faa391 (Li et al. 2016 c\*; Level 2 SOA 15.9 µg m⁻³, Level 3
nitrate 5.16 µg m⁻³), and before that 8753736 (MNFIX split; Level 1 86 nm, Level 3 nitrate
4.96 µg m⁻³). A deployed web app shows the new numbers once it is
redeployed from that branch (its result cache starts empty; prewarm it). The guide counts the 725 species of the MCM file; the notebooks
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
