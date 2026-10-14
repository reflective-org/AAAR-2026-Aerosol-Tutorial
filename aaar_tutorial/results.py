"""Loading saved results and pulling numbers out of a RunResult."""
from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import numpy as np
from tomas_jax.box.result import RunResult

from . import config

# Families of time series in a RunResult, searched in this order by series().
_SERIES_FAMILIES = ("totals", "gases_ppb", "radicals_cm3", "photolysis")
_DIAGNOSTIC_SERIES = ("cs_per_s", "nucleation_rate_cm3_s")


def load_example(preset: str) -> RunResult:
    """Precomputed result of an unmodified preset (no model run needed)."""
    if preset not in config.PRESETS:
        raise ValueError(f"no example for {preset!r}; choose one of {config.PRESETS}")
    text = resources.files("aaar_tutorial").joinpath("examples", f"{preset}.json").read_text(encoding="utf-8")
    return load_result(json.loads(text))


def load_result(source: str | Path | dict) -> RunResult:
    """RunResult from a JSON file written from ``result.to_dict()`` (or from that dict)."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    return as_result(data)


def save_result(result: RunResult, path: str | Path) -> Path:
    """Write ``result`` as JSON (read it back with load_result)."""
    path = Path(path)
    path.write_text(json.dumps(result.to_dict(), separators=(",", ":")), encoding="utf-8")
    return path


def as_result(result: RunResult | dict) -> RunResult:
    """Accept a RunResult or its ``to_dict()`` form; arrays become float numpy arrays."""
    if isinstance(result, RunResult):
        return result
    d = dict(result)

    def arr(x):
        return np.asarray(x, dtype=float)         # JSON null (non-finite) -> nan

    return RunResult(
        scenario=d["scenario"],
        time_h=[float(t) for t in d["time_h"]],
        diameter_nm=arr(d["diameter_nm"]),
        diameter_edges_nm=arr(d["diameter_edges_nm"]),
        dndlog10dp_cm3=arr(d["dndlog10dp_cm3"]),
        dmdlog10dp_ugm3=arr(d["dmdlog10dp_ugm3"]),
        totals={k: arr(v) for k, v in d["totals"].items()},
        gases_ppb={k: arr(v) for k, v in d.get("gases_ppb", {}).items()},
        radicals_cm3={k: arr(v) for k, v in d.get("radicals_cm3", {}).items()},
        h2so4_cm3=arr(d.get("h2so4_cm3", [])),
        photolysis={k: (list(v) if k == "sza_deg" else arr(v))
                    for k, v in d.get("photolysis", {}).items()},
        volatility={k: arr(v) for k, v in d.get("volatility", {}).items()},
        top_contributors=list(d.get("top_contributors", [])),
        diagnostics=dict(d.get("diagnostics", {})),
    )


def series(result: RunResult, key: str) -> np.ndarray:
    """One time series by name, e.g. "SOA_ugm3", "N2O5", "OH", "J_NO2", "h2so4_cm3", "cs_per_s".

    Units as in the RunResult: masses in ug/m3, gases in ppb, radicals and H2SO4 in
    molec/cm3, J in 1/s. Two derived series: "gmd_nm" (number geometric mean diameter) and
    "org_gas_ugm3" (organic vapor summed over all c* classes).
    """
    for family in _SERIES_FAMILIES:
        values = getattr(result, family)
        if key in values:
            return np.asarray(values[key], dtype=float)
    if key == "h2so4_cm3":
        return np.asarray(result.h2so4_cm3, dtype=float)
    if key in _DIAGNOSTIC_SERIES and key in result.diagnostics:
        return np.asarray(result.diagnostics[key], dtype=float)
    if key == "gmd_nm":
        return geometric_mean_diameter_nm(result)
    if key == "org_gas_ugm3" and len(result.volatility.get("gas_ugm3", [])):
        return np.asarray(result.volatility["gas_ugm3"], dtype=float).sum(axis=1)
    raise KeyError(f"{key!r} is not a time series of this run; available: {available_series(result)}")


def available_series(result: RunResult) -> list[str]:
    """Every name series() accepts for this result."""
    names = [k for family in _SERIES_FAMILIES for k in getattr(result, family)]
    names += ["h2so4_cm3"] + [k for k in _DIAGNOSTIC_SERIES if k in result.diagnostics]
    names += ["gmd_nm"] + (["org_gas_ugm3"] if len(result.volatility.get("gas_ugm3", [])) else [])
    return names


def value_at(result: RunResult, key: str, hours: float) -> float:
    """series(result, key) at ``hours`` since the start (linear between outputs)."""
    return float(np.interp(hours, np.asarray(result.time_h, dtype=float), series(result, key)))


def is_night(result: RunResult) -> np.ndarray:
    """True at the output times when the sun is below the horizon (all False without photolysis)."""
    sza = result.photolysis.get("sza_deg")
    if sza is None or len(sza) != len(result.time_h):
        return np.zeros(len(result.time_h), dtype=bool)
    return np.asarray(sza, dtype=float) > config.NIGHT_SZA_DEG


def night_mean(result: RunResult, key: str) -> float:
    """Mean of a series over the night-time outputs (nan if the run has no night)."""
    night = is_night(result)
    return float(series(result, key)[night].mean()) if night.any() else float("nan")


def ugm3_per_ppb(molar_mass: float | str, result_or_scenario) -> float:
    """ug/m3 of 1 ppb of a gas at the run's temperature and pressure.

    ``molar_mass`` in g/mol, or a name from config.MOLAR_MASS_G (e.g. "NO3", "N2O5").
    """
    if isinstance(molar_mass, str):
        if molar_mass not in config.MOLAR_MASS_G:
            raise ValueError(f"no molar mass for {molar_mass!r}; give it in g/mol or use one of "
                             f"{list(config.MOLAR_MASS_G)}")
        molar_mass = config.MOLAR_MASS_G[molar_mass]
    scenario = getattr(result_or_scenario, "scenario", result_or_scenario)
    env = scenario["environment"] if isinstance(scenario, dict) else scenario.environment
    get = env.get if isinstance(env, dict) else lambda k: getattr(env, k)
    mol_per_m3 = get("pres_pa") / (config.GAS_CONSTANT * get("temp_k"))
    return 1e-9 * mol_per_m3 * molar_mass * 1e6


def cstar_at(cstar_298, temp_k):
    """c* (ug/m3) at ``temp_k`` from c* at 298 K, with dHvap = 131 - 11 log10 c* kJ/mol as in the model."""
    cstar_298 = np.asarray(cstar_298, dtype=float)
    a, b = config.DHVAP_KJ
    dh_vap = (a - b * np.log10(cstar_298)) * 1e3                     # J/mol
    return cstar_298 * np.exp(-dh_vap / config.GAS_CONSTANT * (1 / temp_k - 1 / config.CSTAR_T_REF_K))


def has_nitrogen(formula: str) -> bool:
    """True if a molecular formula such as "C10H15NO6" contains N (not Na, Ne, ...)."""
    return bool(re.search(r"N(?![a-z])", formula))


def nitrogen_share(result: RunResult) -> float:
    """Fraction of the top SOA contributors' particle mass in N-containing species (nan if none)."""
    top = result.top_contributors
    total = sum(c["particle_ugm3"] for c in top)
    if not total:
        return float("nan")
    return sum(c["particle_ugm3"] for c in top if has_nitrogen(c["formula"])) / total


def night_intervals(result: RunResult) -> list[tuple[float, float]]:
    """(start, end) in hours of each night (solar zenith angle > 90 deg); [] without photolysis."""
    t = np.asarray(result.time_h, dtype=float)
    night = is_night(result)
    if not night.any():
        return []
    sza = np.asarray(result.photolysis["sza_deg"], dtype=float)

    def crossing(i):
        s0, s1 = sza[i - 1], sza[i]
        if s1 == s0:
            return float(t[i])
        return float(t[i - 1] + (config.NIGHT_SZA_DEG - s0) / (s1 - s0) * (t[i] - t[i - 1]))

    out, start = [], (float(t[0]) if night[0] else None)
    for i in range(1, t.size):
        if night[i] and not night[i - 1]:
            start = crossing(i)
        elif night[i - 1] and not night[i]:
            out.append((start, crossing(i)))
            start = None
    if start is not None:
        out.append((start, float(t[-1])))
    return out


def geometric_mean_diameter_nm(result: RunResult) -> np.ndarray:
    """Number geometric mean dry diameter [nm] at every output time."""
    edges = np.asarray(result.diameter_edges_nm, dtype=float)
    dlog10 = np.diff(np.log10(edges))
    number = np.asarray(result.dndlog10dp_cm3, dtype=float) * dlog10         # [t, bin] #/cm3
    log_d = np.log(np.asarray(result.diameter_nm, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp((number * log_d).sum(axis=1) / number.sum(axis=1))


def summary(result: RunResult) -> dict[str, float]:
    """Headline numbers at the start and end of a run (masses in ug/m3, number in #/cm3)."""
    gmd = geometric_mean_diameter_nm(result)
    out = {"duration_h": float(result.time_h[-1])}
    for key in ("N_cm3", "dry_ugm3", "SOA_ugm3", "SO4_ugm3", "NIT_ugm3", "POA_ugm3"):
        values = series(result, key)
        out[f"{key}_start"] = float(values[0])
        out[f"{key}_end"] = float(values[-1])
    out["gmd_nm_start"] = float(gmd[0])
    out["gmd_nm_end"] = float(gmd[-1])
    return out
