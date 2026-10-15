"""Every constant the tutorial helpers use: pins, URLs, timeouts, cache paths, plot defaults."""
import os
from pathlib import Path

# --- Model pins (keep in step with pyproject.toml and requirements.txt) -------------------
GITHUB_ORG_URL = "https://github.com/reflective-org"
TOMAS_JAX_COMMIT = "87537364420e76cd50203049e4d74cb23660327d"
TUVX_JAX_COMMIT = "e5d9032d22e740d930c578676d726aaa5ba2473a"
VBS_JAX_COMMIT = "4a67f16d5585aee198c62819f1bd52fc0daed9d5"   # pinned by tomas-jax itself
TOMAS_JAX_REQUIREMENT = (
    f"tomas-jax[photolysis] @ git+{GITHUB_ORG_URL}/TOMAS-JAX@{TOMAS_JAX_COMMIT}")

# Versions the presets were tested with (pinned in pyproject.toml where it matters).
TESTED_VERSIONS = {
    "python": "3.12", "jax": "0.11.2", "jaxlib": "0.11.2", "diffrax": "0.7.2",
    "equinox": "0.13.8", "lineax": "0.1.1", "optimistix": "0.1.0", "numpy": "2.5.4",
    "scipy": "1.18.1", "netcdf4": "1.7.4",
}

# This repository. The Colab badges and the install cell of every notebook use TUTORIAL_REF;
# change all of them together with scripts/set_pins.py (tests/test_notebooks.py checks).
TUTORIAL_REPO = "reflective-org/AAAR-2026-Aerosol-Tutorial"
TUTORIAL_REPO_URL = f"https://github.com/{TUTORIAL_REPO}"
TUTORIAL_REF = "main"
TUTORIAL_REQUIREMENT = f"aaar-tutorial @ git+{TUTORIAL_REPO_URL}@{TUTORIAL_REF}"
COLAB_URL = f"https://colab.research.google.com/github/{TUTORIAL_REPO}/blob/{TUTORIAL_REF}/notebooks"
NOTEBOOKS = ("01_volatility.ipynb", "02_limonene_soa.ipynb", "03_day_and_night.ipynb",
             "04_inside_the_model.ipynb")

# Packages whose upgrade by the Colab install cell needs a fresh Python session
# (import name, distribution name).
UPGRADE_SENSITIVE = (("numpy", "numpy"), ("scipy", "scipy"), ("jax", "jax"), ("jaxlib", "jaxlib"),
                     ("ml_dtypes", "ml-dtypes"), ("plotly", "plotly"))

# --- JAX ----------------------------------------------------------------------------------
# The model runs on CPU in float64. Set before jax is imported (ignored if it already is).
JAX_PLATFORMS = "cpu"

# --- TUV-x photolysis data ------------------------------------------------------------------
# tuvx-jax's data/ and examples/ are not in its pip package; the model finds them through
# the TUVX_JAX_ROOT environment variable.
TUVX_ROOT_ENV = "TUVX_JAX_ROOT"
TUVX_TARBALL_URL = f"https://codeload.github.com/reflective-org/tuvx-jax/tar.gz/{TUVX_JAX_COMMIT}"
TUVX_KEEP_DIRS = ("data", "examples")                  # the rest of the tarball is dropped
TUVX_REQUIRED_PATHS = ("examples/tuv_5_4_no_aerosol.json", "data/cross_sections", "data/grids",
                       "data/profiles", "data/quantum_yields")
TUVX_TARBALL_MIN_BYTES = 1_000_000                     # pinned tarball is ~5.2 MB
TUVX_TARBALL_MAX_BYTES = 50_000_000

CACHE_DIR = Path(os.environ.get("AAAR_TUTORIAL_CACHE",
                                Path.home() / ".cache" / "aaar_tutorial"))
TUVX_CACHE_DIR = CACHE_DIR / f"tuvx-jax-{TUVX_JAX_COMMIT}"

DOWNLOAD_TIMEOUT_S = 30.0          # per network operation (connect, each read)
DOWNLOAD_DEADLINE_S = 180.0        # whole download, per attempt
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_RETRY_WAIT_S = (2.0, 5.0)  # pause after attempt 1, 2
DOWNLOAD_CHUNK_BYTES = 1 << 16

# --- Model runs -----------------------------------------------------------------------------
PRESETS = ("level1_volatility", "level2_limonene_soa", "level3_day_night")
RUN_TIMEOUT_S = 900.0              # run() stops a model run after this wall time (None = never)
PROGRESS_MIN_INTERVAL_S = 0.25     # throttle for progress display updates
CSTAR_TABLE_DIR = CACHE_DIR / "cstar_tables"   # c* tables written by run(cstar_ugm3=...)

# Seed-mode keys that run(seed={...}) accepts directly; they change the first seed mode.
SEED_MODE_KEYS = ("number_cm3", "gmd_nm", "gsd")

# --- Notebook tests ---------------------------------------------------------------------------
NOTEBOOK_CELL_TIMEOUT_S = 900      # one cell
NOTEBOOK_TIMEOUT_S = 1800          # one whole notebook

# --- Physical constants and molar masses ------------------------------------------------------
GAS_CONSTANT = 8.314462618         # J mol-1 K-1
BOLTZMANN = 1.380649e-23           # J K-1
AVOGADRO = 6.02214076e23           # mol-1
MOLAR_MASS_G = {"NO3": 62.0, "N2O5": 108.01, "H2SO4": 98.0}   # g/mol, as in tomas_jax.core.config
NIGHT_SZA_DEG = 90.0               # sun below the horizon
CSTAR_T_REF_K = 298.0              # c* inputs are given at this temperature
DHVAP_KJ = (131.0, 11.0)           # dHvap = a - b log10 c*(298 K) [kJ/mol] (Epstein et al., 2010)

# --- Sites (the tutorial web app's Level 3 site picker) -----------------------------------------
# name -> (latitude deg N, longitude deg E, IANA time zone for local clock time)
SITES = {
    "Buffalo, NY": (42.89, -78.88, "America/New_York"),
    "Fort Collins, CO": (40.57, -105.08, "America/Denver"),
    "Pasadena, CA": (34.15, -118.14, "America/Los_Angeles"),
    "Centreville, AL": (32.90, -87.25, "America/Chicago"),
    "Manaus, Brazil": (-3.12, -60.02, "America/Manaus"),
    "Hyytiälä, Finland": (61.85, 24.29, "Europe/Helsinki"),
}
SUNLIGHT_J = ("J_NO2", "J_O3_O1D")  # sunlight_over_day() default
SUNLIGHT_STEP_H = 0.25

# --- Plots ------------------------------------------------------------------------------------
PLOT_TEMPLATE = "plotly_white"
PLOT_HEIGHT = 420
PLOT_HEIGHT_PER_PANEL = 260        # stacked time-series panels
LOG_AXIS_DETAIL_DECADES = 3.0      # log axes spanning fewer decades also get 2 and 5 ticks
LOG_AXIS_LINEAR_BELOW_DECADES = 0.5   # narrower log axes get evenly spaced ticks
PLOT_FONT_SIZE = 13
SLIDER_FRAME_MS = 150              # play-button speed

# Categorical palette of the tomas-web tutorial (light theme), fixed order.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
INITIAL_COLOR = "#9a9a9a"          # t = 0 size distribution
PARTICLE_COLOR = "#008300"
GAS_COLOR = "#2a78d6"
NIGHT_COLOR = "rgba(60, 70, 120, 0.12)"
HEATMAP_COLORSCALE = "Viridis"
HEATMAP_DECADES = 4                # banana plot colour range below its maximum

# Composition stack, bottom to top (keys of RunResult.totals).
COMPOSITION_LAYERS = (
    ("SO4_ugm3", "Sulfate", "#e34948"),
    ("NIT_ugm3", "Nitrate", "#2a78d6"),
    ("NH4_ugm3", "Ammonium", "#eda100"),
    ("POA_ugm3", "POA", "#e87ba4"),
    ("SOA_ugm3", "SOA", "#008300"),
)

DEFAULT_TOTALS = ("N_cm3", "SOA_ugm3", "SO4_ugm3", "NIT_ugm3", "dry_ugm3")
DEFAULT_GASES = {1: (), 2: ("LIMONENE", "O3", "NO", "NO2"), 3: ("O3", "NO2", "N2O5", "HNO3")}
DEFAULT_RADICALS = ("OH", "HO2", "NO3")
DEFAULT_J = ("J_NO2", "J_O3_O1D")

# Axis labels: series key -> (display name, unit).
SERIES_LABELS = {
    "N_cm3": ("Particle number", "cm⁻³"),
    "gmd_nm": ("Geometric mean diameter", "nm"),
    "org_gas_ugm3": ("Organic vapor", "µg m⁻³"),
    "SO4_ugm3": ("Sulfate", "µg m⁻³"),
    "NIT_ugm3": ("Nitrate", "µg m⁻³"),
    "SOA_ugm3": ("SOA", "µg m⁻³"),
    "POA_ugm3": ("POA", "µg m⁻³"),
    "NH4_ugm3": ("Ammonium", "µg m⁻³"),
    "H2O_ugm3": ("Particle water", "µg m⁻³"),
    "dry_ugm3": ("Dry mass", "µg m⁻³"),
    "h2so4_cm3": ("Gas-phase H₂SO₄", "molec cm⁻³"),
    "cs_per_s": ("Condensation sink", "s⁻¹"),
    "nucleation_rate_cm3_s": ("Nucleation rate", "cm⁻³ s⁻¹"),
    "LIMONENE": ("Limonene", "ppb"),
    "O3": ("O₃", "ppb"),
    "NO": ("NO", "ppb"),
    "NO2": ("NO₂", "ppb"),
    "HNO3": ("HNO₃", "ppb"),
    "N2O5": ("N₂O₅", "ppt"),
    "HCHO": ("HCHO", "ppb"),
    "CO": ("CO", "ppb"),
    "SO2": ("SO₂", "ppb"),
    "H2O2": ("H₂O₂", "ppb"),
    "OH": ("OH", "molec cm⁻³"),
    "HO2": ("HO₂", "molec cm⁻³"),
    "NO3": ("NO₃", "molec cm⁻³"),
    "J_NO2": ("J(NO₂)", "s⁻¹"),
    "J_O3_O1D": ("J(O¹D)", "s⁻¹"),
    "J_HCHO_H": ("J(HCHO→H)", "s⁻¹"),
    "J_NO3_NO2": ("J(NO₃→NO₂)", "s⁻¹"),
    "sza_deg": ("Solar zenith angle", "°"),
}
# Series that the time-series plots show in another unit than series() returns: key -> factor.
DISPLAY_SCALE = {"N2O5": 1e3}      # ppb -> ppt
TIME_AXIS_TITLE = "Time since start (h)"
DIAMETER_AXIS_TITLE = "Dry diameter Dp (nm)"
