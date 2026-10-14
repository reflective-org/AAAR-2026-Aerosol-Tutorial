"""Helpers for the AAAR 2026 photochemical box-modeling tutorial notebooks.

    import aaar_tutorial as at
    from aaar_tutorial import plots
    at.setup()                                # TUV-x data + version report
    result = at.run("level1_volatility", surrogate={"cstar_ugm3": 10})
    plots.size_distribution(result)
"""
import os

from . import config
from .install_check import check_session

check_session()                     # before anything below imports numpy, jax or plotly
os.environ.setdefault("JAX_PLATFORMS", config.JAX_PLATFORMS)   # before anything imports jax

from . import plots  # noqa: E402
from .chemistry import mechanism  # noqa: E402
from .data import ensure_photolysis_data  # noqa: E402
from .model import describe, run, scenario_from, setup, versions, write_cstar_table  # noqa: E402
from .results import (available_series, cstar_at, geometric_mean_diameter_nm, is_night,  # noqa: E402
                      load_example, load_result, night_intervals, night_mean, nitrogen_share,
                      save_result, series, summary, ugm3_per_ppb, value_at)
from .sun import site_overrides, sunlight_over_day  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "available_series", "config", "cstar_at", "describe", "ensure_photolysis_data",
    "geometric_mean_diameter_nm", "is_night", "load_example", "load_result", "mechanism",
    "night_intervals", "night_mean", "nitrogen_share", "plots", "run", "save_result",
    "scenario_from", "series", "setup", "site_overrides", "summary", "sunlight_over_day",
    "ugm3_per_ppb", "value_at", "versions", "write_cstar_table",
]
