"""Every constant the tutorial helpers use: pins, URLs, timeouts, cache paths, plot defaults."""
import os
from pathlib import Path

# --- Model pins (keep in step with pyproject.toml and requirements.txt) -------------------
GITHUB_ORG_URL = "https://github.com/reflective-org"
TOMAS_JAX_COMMIT = "0e23da0da79a5a080baa02447e47951b75f796fe"
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
# The look follows the chart cards of the tutorial web app (Figma file jpQseImg7KrgCoMl5tQQyj,
# "X01 tokens and shared rules" and the results frames), on the Reflective brand palette: white
# background; Open Sans, SemiBold titles in Charcoal, secondary text in TEXT_2; left and bottom
# axis lines in Slate, short ticks outside, no grid; thin lines without markers, told apart by
# dash and a label at the line end; units in the subtitle and on the first tick, not in axis
# titles. theme.py builds the Plotly template from these constants.
PLOT_TEMPLATE = "reflective"       # registered in plotly.io.templates by aaar_tutorial.theme
PLOT_HEIGHT = 420
PLOT_HEIGHT_PER_PANEL = 260        # stacked time-series panels
LOG_AXIS_DETAIL_DECADES = 2.2      # log axes spanning fewer decades also get 3 (or 2 and 5) ticks
LOG_AXIS_LINEAR_BELOW_DECADES = 0.5   # narrower log axes get evenly spaced ticks
LOG_AXIS_MAX_TICKS = 6             # wider log axes: every 2nd, 3rd, ... decade, at most this many (incl. ends)
SLIDER_FRAME_MS = 150              # play-button speed
# Pixel sizes. The app's cards are about 480 px wide, a notebook figure about twice that, so the
# sizes are the app's plus 1 px (titles plus 2 px); 10 px is the app's floor.
PLOT_FONT_SIZE = 12                # annotations and other text without its own size
PLOT_TITLE_FONT_SIZE = 15
PLOT_TITLE_FONT_WEIGHT = 600       # Open Sans SemiBold (needs plotly >= 5.23)
PLOT_SUBTITLE_FONT_SIZE = 12
PLOT_TICK_FONT_SIZE = 11
PLOT_LEGEND_FONT_SIZE = 11.5
PLOT_LABEL_FONT_SIZE = 11.5        # labels at line ends, bar values, C_OA label
PLOT_NOTE_FONT_SIZE = 11           # "night", colour-bar ends, captions, slider ticks
PLOT_AXIS_TITLE_FONT_SIZE = 11     # axis titles, kept only where a figure has several y axes
PLOT_PANEL_TITLE_FONT_SIZE = 13    # subplot titles (make_subplots), SemiBold
PLOT_READOUT_FONT_SIZE = 13        # slider readout, JetBrains Mono
PLOT_LINE_WIDTH = 1.8              # px
PLOT_SELECTED_LINE_WIDTH = 2.2     # the snapshot at the slider time
PLOT_COMPARE_LINE_WIDTH = 2.0      # runs A and B
PLOT_AXIS_LINE_WIDTH = 1.0
PLOT_TICK_LENGTH = 3
PLOT_MARKER_SIZE = 7               # only for figures of your own that ask for markers
PLOT_CARD_RADIUS_PX = 10           # card frame round the figure (border LINE)
# Space at the top of every figure (px from the top edge): title, subtitle, legend rows.
PLOT_TITLE_TOP = 12
PLOT_SUBTITLE_SPACE = 18
PLOT_LEGEND_ROW = 20
PLOT_LEGEND_WIDTH_ESTIMATE = 760   # px the legend may fill per row (estimate, sets the margin)

# Reflective brand palette (assets/03_Colors of the brand kit) and the app's tokens (X01 53:370).
# Never use Yellow, Slate, Gold or Green for text: text uses CHARCOAL, TEXT_2 or the *_INK hues.
CHARCOAL = "#282C2E"               # titles, labels, run A
TEXT_2 = "#5A5E61"                 # subtitles, ticks, legends, captions
BLUE = "#466F8D"
GREEN = "#858F6A"
CLAY = "#A0503C"
GOLD = "#A47F52"
SLATE = "#8B8D8E"                  # axis lines and ticks; never text (3.3:1 on white)
SAGE = "#BBC2BD"
BEIGE = "#E1CBBA"
SAND = "#F6EFEA"                   # night bands, slider band, buttons
LINE = "#E8E3DE"                   # card border, label-box border, hover border
GOLD_INK = "#7A5C38"               # text in a Gold series' colour
GREEN_INK = "#5F6748"              # text in a Green series' colour
# Non-brand accents for the 6th to 9th series (as in the presentation's extra model lines).
PLUM, RUST, TEAL, AMBER = "#6B4C8A", "#B5651D", "#3E6B6B", "#E69F00"
AXIS_COLOR = SLATE

# Fonts: Open Sans for all chart text, JetBrains Mono for the slider readout and the species
# codes in plots.top_contributors (as in the app). Most students have neither installed (Colab,
# JupyterHub, most laptops), so the figures load them as web fonts (FONT_CSS_URLS, Google Fonts,
# SIL OFL): widget_iframe() puts the links into its page, and theme.register() adds a script to
# Plotly's HTML renderers (fig.show() in Jupyter and Colab) that loads them and redraws the
# figure once they have arrived. Without network the stacks fall back to installed fonts.
FONT_CSS_URLS = (
    "https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap",
)
FONT_CSS = "".join(f'<link rel="stylesheet" href="{url.replace("&", "&amp;")}">'
                   for url in FONT_CSS_URLS)
# Faces the redraw script waits for (CSS font shorthand); at most FONT_LOAD_TIMEOUT_MS.
FONT_FACES = ('400 16px "Open Sans"', '600 16px "Open Sans"', '700 16px "Open Sans"',
              '400 16px "JetBrains Mono"')
FONT_LOAD_TIMEOUT_MS = 8000
PLOT_FONT_FAMILY = "'Open Sans', 'Helvetica Neue', Arial, sans-serif"
# Chart titles are Open Sans SemiBold, as in the app (a serif only heads its pages: in a
# notebook the markdown headings do that).
PLOT_TITLE_FONT_FAMILY = PLOT_FONT_FAMILY
PLOT_MONO_FONT_FAMILY = "'JetBrains Mono', Menlo, Consolas, monospace"

# plots.partitioning(): volatility classes and the C_OA values on its slider (µg/m³).
PARTITIONING_LOG10_CSTAR = tuple(range(-2, 5))          # c* = 0.01 ... 10 000 µg/m³
# 0.1 (clean air) to 1000 µg/m³ (wildfire smoke), 20 steps per decade, 2 significant digits.
PARTITIONING_COA_UGM3 = tuple(float(f"{10 ** (k / 20):.2g}") for k in range(-20, 61))
PARTITIONING_START_COA_UGM3 = 1.0

# Series colours, in order. Blue, Clay, Green, Charcoal, Gold are the brand hues; Clay comes
# before Green so that SERIES_COLORS[1] ("contains N") differs from PARTICLE_COLOR (Green) in
# plots.top_contributors and notebook 2. Plum, Rust, Teal, Amber follow in the order that keeps
# neighbours apart for colour-blind readers (dataviz validate_palette.js, 2026-10-16). Each series
# also gets its own dash (SERIES_DASHES, in the app's dash vocabulary: solid, 5-3 dashes, dots),
# so lines stay separable without colour. Amber is below 3:1 contrast on white: never text.
SERIES_COLORS = (BLUE, CLAY, GREEN, CHARCOAL, GOLD, PLUM, RUST, TEAL, AMBER)
SERIES_DASHES = ("solid", "5px,3px", "1.5px,2.5px", "10px,3px,2px,3px", "6px,3px,1.5px,3px,1.5px,3px",
                 "12px,4px", "3px,3px", "16px,3px,4px,3px", "2px,5px")
# Markers: none on the plots (the app has none); kept for figures of your own (plots._style).
SERIES_MARKERS = ("circle", "square", "triangle-up", "diamond", "triangle-down", "cross", "x",
                  "hexagon", "star", "pentagon")
# Label colour for a series colour (labels at line ends, peak labels).
INK = {SLATE: TEXT_2, GOLD: GOLD_INK, GREEN: GREEN_INK, AMBER: GOLD_INK, SAGE: GREEN_INK,
       BEIGE: GOLD_INK}
INITIAL_COLOR = SLATE              # t = 0 size distribution (dotted; notebook 1: "grey dotted")
SNAPSHOT_COLOR = BEIGE             # hourly snapshots in plots.size_distribution (faint)
SELECTED_COLOR = BLUE              # the snapshot at the slider time (bold)
PARTICLE_COLOR = GREEN             # notebook 1 text: "green: in the particles, blue: in the gas"
# plots.top_contributors: the "no N" bars (Green) are also hatched, since Green and Clay ("contains
# N") are a red-green pair that colour-blind readers see as close.
NO_N_PATTERN = dict(shape="/", fillmode="overlay", fgcolor="white", fgopacity=0.55, size=7, solidity=0.3)
GAS_COLOR = BLUE
REFERENCE_LINE_COLOR = CHARCOAL    # dashed reference lines (C_OA, SZA = 90°)
REFERENCE_DASH = "4px,3px"
# Comparing two runs (the app's compare view): A Charcoal solid, B Blue dashed.
RUN_A_COLOR, RUN_B_COLOR = CHARCOAL, BLUE
RUN_B_DASH = "6px,3px"
# Night bands: Sand, with the word "night" at the top. Beige at 30 % gives Sand on white and stays
# see-through, so a band drawn over lines (e.g. by add_vrect's default layer) does not hide them.
NIGHT_COLOR = "rgba(225, 203, 186, 0.3)"
NIGHT_LABEL = "night"
# composition(fraction=True): the 100 % stack fills the plot and would hide a band behind it, so
# there the night band is drawn over the stack, Sand at 30 %, and "night" goes above the plot.
NIGHT_OVERLAY_COLOR = "rgba(246, 239, 234, 0.3)"
NIGHT_LABEL_CLEARANCE_PX = 6       # least space between the "night" label and a line
COMPOSITION_FILL_OPACITY = 0.85    # stacked areas: the night bands behind them show faintly
# Banana plot: ColorBrewer YlGnBu (pale yellow -> green -> blue -> navy; lightness falls
# monotonically, colour-blind safe), with white below it, so empty and near-empty cells blend into
# the white background (Ali's choice, 2026-10-16, over a one-hue blue ramp, inferno, YlOrBr, a
# Reflective-brand ramp and the app's beige-to-navy steps).
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
HEATMAP_COLORBAR_PX = (100, 8)     # small horizontal colour bar at the top right (length, height)
SCALE_BUTTON_LABELS = ("Linear", "Log")   # scale buttons on the plots (top right)
HEATMAP_DECADES = 4                # banana plot colour range below its maximum

# Composition stack, bottom to top (keys of RunResult.totals). Brand versions of the AMS
# convention (sulfate red, nitrate blue, ammonium orange, organics green); POA light Sage under
# the SOA Green, so the two organic layers read as one family and SOA has the Green of "particle
# phase" in the volatility plots (the app uses slate, clay, green and gold and has no ammonium).
# White lines separate the layers.
COMPOSITION_LAYERS = (
    ("SO4_ugm3", "Sulfate", CLAY),
    ("NIT_ugm3", "Nitrate", BLUE),
    ("NH4_ugm3", "Ammonium", GOLD),
    ("POA_ugm3", "POA", SAGE),
    ("SOA_ugm3", "SOA", GREEN),
)

# Line colour of each quantity in a single-run time series (one panel per quantity): the
# composition colours, so SOA is Green and sulfate Clay in totals() as in composition(); totals
# in Charcoal (Blue is nitrate's); nucleation rate Clay and condensation sink Green as in the app;
# gases and radicals as in SPECIES_STYLE; any other quantity SERIES_COLORS[0].
QUANTITY_COLORS = {key: color for key, _, color in COMPOSITION_LAYERS}
QUANTITY_COLORS.update({"N_cm3": CHARCOAL, "dry_ugm3": CHARCOAL, "nucleation_rate_cm3_s": CLAY,
                        "cs_per_s": GREEN})

# Gases, radicals and photolysis frequencies: one colour and dash per species, as in the app
# (X01 81:117). NO2 held constant (scenario "hold_constant") is drawn Slate dashed instead.
SPECIES_STYLE = {
    "OH": (BLUE, "solid"), "HO2": (GREEN, "5px,3px"), "NO3": (CLAY, "1.5px,2.5px"),
    "O3": (BLUE, "solid"), "NO2": (CLAY, "5px,3px"), "HNO3": (GOLD, "5px,3px"),
    "N2O5": (CLAY, "solid"), "LIMONENE": (GREEN, "solid"), "NO": (SLATE, "1.5px,2.5px"),
    "J_NO2": (BLUE, "solid"), "J_O3_O1D": (GOLD, "5px,3px"),
}
HELD_STYLE = (SLATE, "4px,3px")
# Species the app does not style, in this order (pairs already used in a plot are skipped).
SPECIES_FALLBACK_STYLES = ((PLUM, "solid"), (TEAL, "5px,3px"), (RUST, "1.5px,2.5px"),
                           (CHARCOAL, "10px,3px,2px,3px"), (PLUM, "5px,3px"), (TEAL, "solid"),
                           (RUST, "5px,3px"), (CHARCOAL, "3px,3px"))

DEFAULT_TOTALS = ("N_cm3", "SOA_ugm3", "SO4_ugm3", "NIT_ugm3", "dry_ugm3")
DEFAULT_GASES = {1: (), 2: ("LIMONENE", "O3", "NO", "NO2"), 3: ("O3", "NO2", "N2O5", "HNO3")}
DEFAULT_RADICALS = ("OH", "HO2", "NO3")
DEFAULT_J = ("J_NO2", "J_O3_O1D")

# Axis labels: series key -> (display name, unit). Units written as in the app (µg/m³).
SERIES_LABELS = {
    "N_cm3": ("Particle number", "cm⁻³"),
    "gmd_nm": ("Geometric mean diameter", "nm"),
    "org_gas_ugm3": ("Organic vapor", "µg/m³"),
    "SO4_ugm3": ("Sulfate", "µg/m³"),
    "NIT_ugm3": ("Nitrate", "µg/m³"),
    "SOA_ugm3": ("SOA", "µg/m³"),
    "POA_ugm3": ("POA", "µg/m³"),
    "NH4_ugm3": ("Ammonium", "µg/m³"),
    "H2O_ugm3": ("Particle water", "µg/m³"),
    "dry_ugm3": ("Dry mass", "µg/m³"),
    "h2so4_cm3": ("Gas-phase H₂SO₄", "molec/cm³"),
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
    "OH": ("OH", "molec/cm³"),
    "HO2": ("HO₂", "molec/cm³"),
    "NO3": ("NO₃", "molec/cm³"),
    "J_NO2": ("J(NO₂)", "s⁻¹"),
    "J_O3_O1D": ("J(O¹D)", "s⁻¹"),
    "J_HCHO_H": ("J(HCHO→H)", "s⁻¹"),
    "J_NO3_NO2": ("J(NO₃→NO₂)", "s⁻¹"),
    "sza_deg": ("Solar zenith angle", "°"),
}
# Series that the time-series plots show in another unit than series() returns: key -> factor.
DISPLAY_SCALE = {"N2O5": 1e3}      # ppb -> ppt
# Axis titles for figures of your own (the plots put units in the subtitle and on the ticks).
TIME_AXIS_TITLE = "Time since start (h)"
DIAMETER_AXIS_TITLE = "Dry diameter Dp (nm)"
