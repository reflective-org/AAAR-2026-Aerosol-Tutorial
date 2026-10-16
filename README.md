# AAAR 2026 Aerosol Box-Modeling Tutorial

Jupyter notebooks for the AAAR 2026 tutorial on photochemical box modeling (instructor: Shantanu
Jathar, Colorado State University). They run [TOMAS-JAX](https://github.com/reflective-org/TOMAS-JAX),
a box model that couples explicit gas-phase chemistry (the MCM v3.3.1 limonene mechanism), TUV-x
photolysis and TOMAS sectional aerosol microphysics with kinetic SOA partitioning.

The tutorial web app teaches the same three levels with sliders. These notebooks are the "look
under the hood" track: you import the model, change any input from Python, run it and explore the
results in interactive Plotly figures. You need basic aerosol and atmospheric chemistry; the Python
is explained as you go.

## Notebooks

| | Notebook | What you learn | Run time (laptop) |
|---|---|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reflective-org/AAAR-2026-Aerosol-Tutorial/blob/main/notebooks/01_volatility.ipynb) | [Level 1 · Volatility](notebooks/01_volatility.ipynb) | Saturation concentration c\*, absorptive partitioning, condensation of a semivolatile vs a non-volatile vapor (H₂SO₄), coagulation, nucleation. One made-up organic vapor, no gas chemistry. | about 25 s |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reflective-org/AAAR-2026-Aerosol-Tutorial/blob/main/notebooks/02_limonene_soa.ipynb) | [Level 2 · Limonene SOA](notebooks/02_limonene_soa.ipynb) | An explicit mechanism, which oxidant removes limonene, how products get a c\* and are lumped into volatility classes, the top SOA species, and NOx, O₃, limonene and seed experiments. Constant chamber-like light. | about 1.5 min |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reflective-org/AAAR-2026-Aerosol-Tutorial/blob/main/notebooks/03_day_and_night.ipynb) | [Level 3 · Day & Night](notebooks/03_day_and_night.ipynb) | Solar zenith angle and photolysis rates for a site and date, the day–night cycle of OH and NO₃, N₂O₅ uptake (γ) making particle nitrate, sites and seasons, and what holding NO₂ constant does. | about 2 min |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reflective-org/AAAR-2026-Aerosol-Tutorial/blob/main/notebooks/04_inside_the_model.ipynb) | [Inside the model](notebooks/04_inside_the_model.ipynb) | TUV-x vs MCM photolysis, from a KPP mechanism to stiff ODEs and the solver, Pankow partitioning in the model code, N₂O₅ uptake from γ and surface area, and the operator-splitting loop written out. | under 1 min |

Run times are for running every cell once on a laptop (Apple Silicon), after installation, with
the default 40 size bins (more bins are slower). Colab and older laptops are about twice as slow.
The first model run in each session also compiles the model with JAX (about 2 s for Level 1 and
5 s for Levels 2 and 3 on a laptop).

The Colab badges open the notebooks from `main`. All work happens on `main`, so every push reaches
students straight away: run the tests before pushing (see the end of this README).

## Run on Colab

1. Click a badge above. Each notebook opens in its own Colab runtime (CPU is enough).
2. *Runtime › Run all*, or run the cells one by one with Shift+Enter.
3. The first code cell installs the model and the tutorial helpers (1–2 minutes). Red pip
   "dependency conflicts" lines are expected on Colab and do not matter here.
4. If a cell asks you to restart, click *Runtime › Restart session*, then *Runtime › Run all*
   again: after the first install, or after the helpers were updated while a runtime was open.

To keep your changes, use *File › Save a copy in Drive*.

## Run on your own computer

You need Python 3.12 or newer and git. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/reflective-org/AAAR-2026-Aerosol-Tutorial.git
cd AAAR-2026-Aerosol-Tutorial
uv sync --extra dev                 # model, helpers, JupyterLab and the test tools
uv run jupyter lab notebooks/
```

With pip:

```bash
git clone https://github.com/reflective-org/AAAR-2026-Aerosol-Tutorial.git
cd AAAR-2026-Aerosol-Tutorial
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
jupyter lab notebooks/
```

The install pulls the pinned model versions from GitHub (TOMAS-JAX, tuvx-jax, vbs-jax), so it needs
internet access once. The first `at.setup()` downloads the TUV-x photolysis data (about 5 MB) into
`~/.cache/aaar_tutorial` (set `AAAR_TUTORIAL_CACHE` to put it elsewhere). VS Code works too: open a
notebook and pick the `.venv` kernel.

## Troubleshooting

- **The first model run is slow.** JAX compiles the model the first time it runs in a session (about
  2 s for Level 1 and 5 s for Levels 2 and 3 on a laptop, about twice that on Colab), and again
  when a run switches a process on or off. Later runs are fast. `at.setup(warmup=True)` compiles
  every preset up front.
- **"Installed, but this Python session still has the old versions loaded".** The install cell
  upgraded numpy, jax or plotly after Python had loaded the old version. Restart the session
  (Colab: *Runtime › Restart session*) and run the notebook again from the top.
- **`ModuleNotFoundError: This notebook needs the tutorial helpers ...`** on your own computer:
  pick the kernel of the repository's `.venv` (the usual cause of "found none"), or update the
  repository (`git pull`, then `uv sync` or `pip install -e .`), then restart the kernel.
- **"This session still has older tutorial helpers loaded"** or **`AttributeError: module
  'aaar_tutorial.plots' has no attribute ...`**: the helpers were updated after this session imported
  them. Colab: *Runtime › Restart session*, then *Runtime › Run all* (the install cell updates the
  helpers first). Locally: `git pull`, `uv sync`, then *Kernel › Restart Kernel*.
- **A run is slow or seems stuck.** Colab is about twice as slow as a laptop. Every run stops with a
  `TimeoutError` after 15 minutes (`at.run(..., timeout_s=None)` turns the limit off). Notebooks 1–3
  have a `USE_PRECOMPUTED = True` switch that loads a stored result of the preset instead of
  running it.
- **"unknown input 'cstar'. Did you mean ...":** inputs are grouped as in the model's `Scenario`, e.g.
  `surrogate={"cstar_ugm3": 10}` or `environment={"rh": 0.8}`. `at.describe("level1_volatility")`
  lists every input of a preset.
- **TUV-x download fails.** It tries three times with time limits. Behind a firewall, download
  <https://codeload.github.com/reflective-org/tuvx-jax/tar.gz/e5d9032d22e740d930c578676d726aaa5ba2473a>,
  unpack it and set `TUVX_JAX_ROOT` to the folder that contains `data/` and `examples/`.
- **Plots do not show on GitHub.** GitHub renders notebooks without their interactive output, and the
  saved notebooks have no outputs. Open them in Colab or Jupyter.

## The model and its assumptions

**[How the model works](docs/model_flowchart.pdf)**: a two-page flowchart of one coupling interval (chemistry,
hand-off to the aerosol, hand-back) and of one aerosol micro-step, showing where particles move between size
bins (PPM) and what each level switches on.

The notebooks pin every model package to a commit, so the results do not change under you:

- [TOMAS-JAX](https://github.com/reflective-org/TOMAS-JAX/tree/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c)
  (the box model; pinned in `pyproject.toml`). Its tutorial documentation:
  [gas chemistry](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/docs/aaar/gaschem.md),
  [coupling](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/docs/aaar/coupling.md),
  [aerosol step](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/docs/aaar/aerosol_step.md),
  [SOA partitioning](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/docs/aaar/soa_partitioning.md),
  [N₂O₅ uptake](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/docs/aaar/heterogeneous.md).
- [tuvx-jax](https://github.com/reflective-org/tuvx-jax) (TUV-x photolysis, commit `e5d9032`) and
  [vbs-jax](https://github.com/reflective-org/vbs-jax) (kinetic SOA partitioning, commit `4a67f16`).
- **[ASSUMPTIONS.md](https://github.com/reflective-org/TOMAS-JAX/blob/1f519e910b5e3119a54ce3df1617aa32a4d5ab2c/ASSUMPTIONS.md)
  lists every simplification.** Two matter most for interpreting results:
  - The c\* of the Level 2 and 3 products comes from a composition-only formula (Li et al., 2016) as a
    placeholder until a curated per-species table arrives (A8). With it, about 98 % of the Level 2
    SOA contains nitrogen (organic nitrates and PANs), and without NOx there is almost no SOA. Treat
    Level 2 SOA amounts and yields as a lesson in how much they depend on c\*, not as limonene facts.
  - Level 2 light is constant (A10). Level 3 holds NO₂ at 1 ppb (reset after every 15-minute interval) as a stand-in for NOx emissions, and
    N₂O₅ uptake (γ) is the only route to particle nitrate (A1, A2, A5, A14).

## What is in this repository

```text
notebooks/          the four tutorial notebooks (saved without outputs)
aaar_tutorial/      helpers the notebooks import: run(), plots, setup(), TUV-x data download
  config.py         every constant: pins, URLs, timeouts, sites, plot defaults
  examples/         stored results of the three presets (instant plots, no model run)
tests/              unit tests and notebook checks (pytest)
scripts/set_pins.py changes the tutorial ref or a model commit everywhere at once
docs/INSTRUCTORS.md session plan, timings, caveats, how to update pins
docs/model_flowchart.pdf  how the model works, in two pages (source: docs/flowchart/)
```

Tests: `uv run pytest` (about 10 s), and `uv run pytest -m notebooks` runs every notebook top to
bottom (about 4 minutes). GitHub Actions runs both on every push.

Instructors: see [docs/INSTRUCTORS.md](docs/INSTRUCTORS.md).

## Licence

MIT, see [LICENSE](LICENSE). The model packages the notebooks install have their own licences
(TOMAS-JAX and vbs-jax: LGPL-3.0).
