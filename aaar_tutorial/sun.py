"""Where and when the box is: site overrides and the sun over a day (no chemistry)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from . import config


def site_overrides(site: str, day, local_hour: int = 12) -> dict:
    """run() inputs that put the box at ``site`` (a key of config.SITES), starting at
    ``local_hour`` local clock time on ``day`` (a date or "YYYY-MM-DD")."""
    if site not in config.SITES:
        raise ValueError(f"unknown site {site!r}; choose one of {list(config.SITES)} "
                         "or add it to aaar_tutorial.config.SITES")
    lat, lon, tz_name = config.SITES[site]
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:          # no time-zone database (e.g. Windows): solar time
        tz = timezone(timedelta(hours=round(lon / 15)))
    day = date.fromisoformat(day) if isinstance(day, str) else day
    start = datetime(day.year, day.month, day.day, local_hour, tzinfo=tz)
    return {"start_utc": start.astimezone(timezone.utc).isoformat(),
            "photolysis": {"latitude_deg": lat, "longitude_deg": lon}}


def sunlight_over_day(site: str, day, local_hour: int = 12, hours: float = 24.0,
                      j=config.SUNLIGHT_J, step_h: float = config.SUNLIGHT_STEP_H) -> dict:
    """Solar zenith angle and photolysis frequencies straight from TUV-x, without chemistry.

    Returns {"label", "local_start_h", "time_h", "sza_deg", <J name>: array (1/s), ...}.
    The first call in a session builds the TUV-x table (a few seconds).
    """
    from tomas_jax.gaschem.photolysis import JProvider, PhotolysisSettings

    from .data import ensure_photolysis_data

    ensure_photolysis_data(verbose=False)
    j = [j] if isinstance(j, str) else list(j)
    o = site_overrides(site, day, local_hour)
    provider = JProvider(j, PhotolysisSettings(
        "tuvx", o["photolysis"]["latitude_deg"], o["photolysis"]["longitude_deg"],
        datetime.fromisoformat(o["start_utc"])))
    t_h = np.arange(0.0, hours + 1e-9, step_h)
    values = np.array([provider.at(3600.0 * t) for t in t_h])
    out = {"label": f"{site}, {day}", "local_start_h": local_hour, "time_h": t_h,
           "sza_deg": np.array([provider.sza_deg(3600.0 * t) for t in t_h])}
    out.update({name: values[:, n] for n, name in enumerate(j)})
    return out
