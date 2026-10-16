"""Every constant the tutorial helpers use: pins, URLs, timeouts, cache paths, plot defaults."""
import os
from pathlib import Path

# --- Model pins (keep in step with pyproject.toml and requirements.txt) -------------------
GITHUB_ORG_URL = "https://github.com/reflective-org"
TOMAS_JAX_COMMIT = "2faa391bcf132fb6796dc71bb637974f9282a459"
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
# Images in the notebooks: header banners and the "Supported by Reflective" badge
# (scripts/make_banners.py). Linked at the commit that added them, so the URLs work on every
# branch and in Colab and never change. After regenerating the images, commit them first, then
# set ASSET_COMMIT to that commit and update the notebook links (tests/test_notebooks.py checks).
ASSET_COMMIT = "9feba6889d498d8c754a6783e5460f46e054d60f"
ASSET_URL = f"https://raw.githubusercontent.com/{TUTORIAL_REPO}/{ASSET_COMMIT}/assets"
REFLECTIVE_URL = "https://www.reflective.org"
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
# Size bins for runs started from a preset name (at.use_bins() changes it for the session,
# run(numerics={"nbins": ...}) for one run). Same diameter range, 1.7 nm to 17.5 µm, for every
# choice. With the PPM bin transfer (TOMAS-JAX 1f519e9) the presets' 40 bins are within 0.6 % of
# 160-320 bins in mean diameter, SOA and nitrate, so 40 (also the web app's grid) is the default;
# more bins draw finer curves. Measured 2026-10-16, laptop, repeat runs, at 40 / 80 / 160 / 320
# bins: Level 1 1.1 / 4.1 / 14 / 61 s, Level 3 12 / 19 / 39 / 136 s; Colab about twice as slow.
NBINS = 40
NBINS_CHOICES = (40, 80, 160, 320)
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
# The look follows Reflective's presentation figures (Ali's comparison/figures/style.py): white
# background, Charcoal text and axis lines, left and bottom axes only, faint grid, frameless
# legend, thick lines, brand colours, and per-series dash and marker patterns. theme.py builds
# the Plotly template from these constants.
PLOT_TEMPLATE = "reflective"       # registered in plotly.io.templates by aaar_tutorial.theme
PLOT_HEIGHT = 420
PLOT_HEIGHT_PER_PANEL = 260        # stacked time-series panels
LOG_AXIS_DETAIL_DECADES = 3.0      # log axes spanning fewer decades also get 2 and 5 ticks
LOG_AXIS_LINEAR_BELOW_DECADES = 0.5   # narrower log axes get evenly spaced ticks
PLOT_FONT_SIZE = 13
SLIDER_FRAME_MS = 150              # play-button speed

# Reflective brand palette (assets/03_Colors of the brand kit). Never use Yellow for text.
CHARCOAL = "#282C2E"               # all text, axis lines and ticks
BLUE = "#466F8D"
GREEN = "#858F6A"
CLAY = "#A0503C"
GOLD = "#A47F52"
SLATE = "#8B8D8E"
SAGE = "#BBC2BD"
BEIGE = "#E1CBBA"
SAND = "#F5EFEA"                   # Beige lightened: button and slider fills
# Non-brand accents for the 6th to 9th series (as in the presentation's extra model lines).
PLUM, RUST, TEAL, AMBER = "#6B4C8A", "#B5651D", "#3E6B6B", "#E69F00"

# Fonts: Open Sans (body) and Gambarino (display, titles). Most students have neither installed
# (Colab, JupyterHub, most laptops), so the figures load them as web fonts (FONT_CSS_URLS):
# widget_iframe() puts the links into its page, and theme.register() adds a script to Plotly's
# HTML renderers (fig.show() in Jupyter and Colab) that loads them and redraws the figure once
# they have arrived. Open Sans and Libre Caslon Text: Google Fonts (SIL OFL). Gambarino:
# Fontshare (ITF Free Font License: free for personal and commercial use in any medium incl.
# web, served from ITF's own servers through the Fontshare API; the font files themselves must
# not be redistributed, so they are linked, never copied into this repository). Without network
# the stacks fall back to installed fonts; the title stack tries serifs with lining figures
# before Georgia, whose old-style figures make "1" look like "ı".
FONT_CSS_URLS = (
    "https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700"
    "&family=Libre+Caslon+Text&display=swap",
    "https://api.fontshare.com/v2/css?f[]=gambarino@400&display=swap",
)
FONT_CSS = "".join(f'<link rel="stylesheet" href="{url.replace("&", "&amp;")}">'
                   for url in FONT_CSS_URLS)
# Faces the redraw script waits for (CSS font shorthand); at most FONT_LOAD_TIMEOUT_MS.
FONT_FACES = ('400 16px "Open Sans"', '600 16px "Open Sans"', '700 16px "Open Sans"',
              '400 16px Gambarino')
FONT_LOAD_TIMEOUT_MS = 8000
PLOT_FONT_FAMILY = "'Open Sans', 'Helvetica Neue', Arial, sans-serif"
PLOT_TITLE_FONT_FAMILY = ("Gambarino, 'Libre Caslon Text', 'Iowan Old Style', 'Palatino Linotype', "
                          "Palatino, Georgia, serif")
PLOT_PANEL_TITLE_FONT_SIZE = 17    # subplot titles (make_subplots), in the title font
PLOT_SUBTITLE_SPACE = 22           # px of top margin added for a subtitle (plots.style)
PLOT_TITLE_FONT_SIZE = 21
PLOT_LINE_WIDTH = 2.5              # px
PLOT_AXIS_LINE_WIDTH = 1.2
PLOT_GRID_COLOR = "rgba(40, 44, 46, 0.08)"
PLOT_MARKER_SIZE = 7
PLOT_MARKERS_PER_LINE = 10         # markers drawn per line (marker.maxdisplayed): sparse, staggered

# plots.partitioning(): volatility classes and the C_OA values on its slider (µg/m³).
PARTITIONING_LOG10_CSTAR = tuple(range(-2, 5))          # c* = 0.01 ... 10 000 µg/m³
# 0.1 (clean air) to 1000 µg/m³ (wildfire smoke), 20 steps per decade, 2 significant digits.
PARTITIONING_COA_UGM3 = tuple(float(f"{10 ** (k / 20):.2g}") for k in range(-20, 61))
PARTITIONING_START_COA_UGM3 = 1.0

# Series colours, in order. Blue, Clay, Green, Charcoal, Gold are the brand hues; Clay comes
# before Green so that SERIES_COLORS[1] ("contains N") differs from PARTICLE_COLOR (Green) in
# plots.top_contributors and notebook 2. Plum, Rust, Teal, Amber follow in the order that keeps
# neighbours apart for colour-blind readers (dataviz validate_palette.js, 2026-10-16). Each series
# also gets its own dash and marker (SERIES_DASHES, SERIES_MARKERS), so lines stay separable
# without colour. Amber is below 3:1 contrast on white: always drawn with its marker and dash.
SERIES_COLORS = (BLUE, CLAY, GREEN, CHARCOAL, GOLD, PLUM, RUST, TEAL, AMBER)
SERIES_DASHES = ("solid", "dashdot", "dot", "16px,5px", "9px,3px,3px,3px,3px,3px", "longdashdot",
                 "6px,4px", "20px,4px,6px,4px", "3px,4px,3px,4px,12px,4px")
# One marker more than colours and dashes: the (colour, dash, marker) triple repeats only after
# 90 series (the 10th gas would otherwise look exactly like the 1st).
SERIES_MARKERS = ("circle", "square", "triangle-up", "diamond", "triangle-down", "cross", "x",
                  "hexagon", "star", "pentagon")
INITIAL_COLOR = SLATE              # t = 0 size distribution (dotted)
PARTICLE_COLOR = GREEN             # notebook 1 text: "green: in the particles, blue: in the gas"
GAS_COLOR = BLUE
REFERENCE_LINE_COLOR = CHARCOAL    # dashed reference lines (C_OA, SZA = 90°)
NIGHT_COLOR = "rgba(139, 141, 142, 0.16)"   # Slate, light: night bands
# Night bands drawn over the stacked composition areas (which would hide NIGHT_COLOR below them).
NIGHT_OVERLAY_COLOR = "rgba(139, 141, 142, 0.10)"
# Banana plot: ColorBrewer YlGnBu (pale yellow -> green -> blue -> navy; lightness falls
# monotonically, colour-blind safe), with white below it, so empty and near-empty cells blend into
# the white background (Ali's choice, 2026-10-16, over a one-hue blue ramp, inferno, YlOrBr and a
# Reflective-brand ramp).
HEATMAP_COLORSCALE = [
    [0.0, "#ffffff"],
    [0.02, "#ffffd9"],
    [0.1425, "#edf8b1"],
    [0.265, "#c7e9b4"],
    [0.3875, "#7fcdbb"],
    [0.51, "#41b6c4"],
    [0.6325, "#1d91c0"],
    [0.755, "#225ea8"],
    [0.8775, "#253494"],
    [1.0, "#081d58"],
]
SCALE_BUTTON_LABELS = ("Linear", "Log")   # scale buttons on the plots (top right)
HEATMAP_DECADES = 4                # banana plot colour range below its maximum

# Composition stack, bottom to top (keys of RunResult.totals). Brand versions of the AMS
# convention (sulfate red, nitrate blue, ammonium orange, organics green); POA light Sage under
# the SOA Green, so the two organic layers read as one family. White lines separate the layers.
COMPOSITION_LAYERS = (
    ("SO4_ugm3", "Sulfate", CLAY),
    ("NIT_ugm3", "Nitrate", BLUE),
    ("NH4_ugm3", "Ammonium", GOLD),
    ("POA_ugm3", "POA", SAGE),
    ("SOA_ugm3", "SOA", GREEN),
)

# Line colour of each quantity in a single-run time series (one panel per quantity): the
# composition colours, so SOA is Green and sulfate Clay in totals() as in composition(); totals
# in Charcoal; any other quantity SERIES_COLORS[0].
QUANTITY_COLORS = {key: color for key, _, color in COMPOSITION_LAYERS}
QUANTITY_COLORS.update({"N_cm3": CHARCOAL, "dry_ugm3": CHARCOAL})

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
