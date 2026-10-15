"""Thin wrappers around the TOMAS-JAX box model: session setup and runs with a progress bar.

The model itself is ``tomas_jax.box`` (``scenario.load_preset``, ``Scenario.replace``,
``driver.run``); these helpers only add data download, input checks, progress display,
custom c* tables and a timeout.
"""
from __future__ import annotations

import contextlib
import dataclasses
import difflib
import hashlib
import json
import platform
import time
from importlib import metadata
from unittest import mock

from . import config
from .data import ensure_photolysis_data
from .install_check import check_session


def setup(warmup: bool = False, verbose: bool = True) -> None:
    """Prepare this session: check the install, fetch TUV-x data, print versions, optionally compile."""
    check_session()
    root = ensure_photolysis_data(verbose=verbose)
    if verbose:
        info = {**versions(), "tuvx_data": str(root)}
        width = max(len(k) for k in info)
        for key, value in info.items():
            print(f"{key:<{width}}  {value}")
        if not info["tomas-jax"].startswith(config.TOMAS_JAX_COMMIT[:7]):
            print(f"Note: these helpers expect tomas-jax {config.TOMAS_JAX_COMMIT[:7]}. On Colab, "
                  "Runtime > Disconnect and delete runtime, then Run all, installs it fresh.")
    if warmup:
        from tomas_jax.box import driver
        t0 = time.perf_counter()
        driver.warmup()
        if verbose:
            print(f"Model compiled for all presets in {time.perf_counter() - t0:.1f} s")


def versions() -> dict[str, str]:
    """Python, JAX and model package versions (git packages show their commit)."""
    import jax
    import tomas_jax                     # noqa: F401  (switches JAX to float64)

    info = {"python": platform.python_version(), "jax": jax.__version__,
            "jax backend": f"{jax.default_backend()} (float64: {jax.config.jax_enable_x64})"}
    for dist in ("tomas-jax", "tuvx-jax", "vbs-jax"):
        info[dist] = _source_of(dist)
    return info


def run(scenario, *, progress=True, timeout_s: float | None = config.RUN_TIMEOUT_S,
        cstar_ugm3: dict[str, float] | None = None, **overrides):
    """Run the box model and return a RunResult.

    ``scenario`` is a preset name (e.g. "level2_limonene_soa"), a Scenario, a RunResult
    (its scenario is rerun) or a scenario dict. Keyword overrides go to
    ``Scenario.replace``: nested sections take dicts (``environment={"rh": 0.8}``),
    ``gases_ppb`` is merged into the preset's gases (other species keep their values) and
    ``seed`` also accepts ``number_cm3``, ``gmd_nm`` and ``gsd`` of the first seed mode.
    Use ``name=...`` to label the run in comparison plots.

    ``progress``: True shows a progress bar, False nothing; a function is called with the
    finished fraction (0 to 1), e.g. to drive an ipywidgets progress bar.
    ``cstar_ugm3``: {species: c* at 298 K in ug/m3} replaces the Li et al. (2016) estimate
    for those species (Levels 2 and 3).
    """
    from tomas_jax.box import driver

    if cstar_ugm3 is not None:
        overrides["cstar_table"] = write_cstar_table(cstar_ugm3)
    s = scenario_from(scenario, **overrides)
    if s.cstar_table and not s.processes.chemistry:
        raise ValueError("a c* table needs gas chemistry (Level 2 or 3); Level 1 sets "
                         "surrogate={'cstar_ugm3': ...} instead")
    if s.processes.chemistry and s.photolysis.mode in ("tuvx", "constant"):
        ensure_photolysis_data()
    bar = None
    if callable(progress):
        progress_cb = progress
    elif progress:
        bar = _Progress(f"{s.name} ({s.duration_h:g} h)")
        progress_cb = bar.update
    else:
        progress_cb = None
    cancel = None
    if timeout_s is not None:
        deadline = time.monotonic() + timeout_s
        cancel = lambda: time.monotonic() > deadline          # noqa: E731
    try:
        with _cstar_table_visible(s.cstar_table):
            result = driver.run(s, progress_cb=progress_cb, cancel=cancel)
    except BaseException as err:
        if bar:
            bar.finish("stopped")
        if isinstance(err, RuntimeError) and str(err) == "run cancelled":
            raise TimeoutError(f"run stopped after timeout_s={timeout_s:g} s; pass a larger "
                               "timeout_s, or timeout_s=None for no limit") from None
        raise
    if bar:
        bar.finish(f"done in {result.diagnostics['runtime_s']:.1f} s")
    for warning in result.diagnostics.get("warnings", []):
        print(f"Warning: {warning}")
    return result


def scenario_from(scenario, **overrides):
    """Scenario from a preset name, Scenario, RunResult or scenario dict, with run()'s overrides.

    A preset name gets ``config.NBINS`` size bins (the presets themselves have 40); a Scenario,
    RunResult or dict keeps its own. ``numerics={"nbins": ...}`` overrides either.
    """
    from tomas_jax.box.result import RunResult
    from tomas_jax.box.scenario import Scenario, from_dict, list_presets, load_preset

    if isinstance(scenario, RunResult):
        scenario = scenario.scenario
    if isinstance(scenario, dict):
        s = from_dict(scenario)
    elif isinstance(scenario, str):
        if scenario not in list_presets():
            raise ValueError(f"unknown preset {scenario!r}; choose one of {list_presets()}")
        s = load_preset(scenario).replace(numerics={"nbins": config.NBINS})
    elif isinstance(scenario, Scenario):
        s = scenario
    else:
        raise TypeError(f"expected a preset name, Scenario, RunResult or dict, "
                        f"got {type(scenario).__name__}")
    if not overrides:
        return s
    _check_inputs(overrides)
    if isinstance(overrides.get("gases_ppb"), dict):
        overrides["gases_ppb"] = {**s.gases_ppb, **overrides["gases_ppb"]}
    if isinstance(overrides.get("seed"), dict):
        overrides["seed"] = _expand_seed(s.seed, overrides["seed"])
    return s.replace(**overrides)


def write_cstar_table(cstar_ugm3: dict[str, float]) -> str:
    """Write {species: c* at 298 K [ug/m3]} as a c* table and return its file name.

    The name contains a hash of the contents, because the model caches tables by name.
    """
    rows = "".join(f"{name},{float(value):.6g}\n" for name, value in sorted(cstar_ugm3.items()))
    text = "species,cstar_ugm3\n" + rows
    name = f"cstar_{hashlib.sha1(text.encode()).hexdigest()[:12]}.csv"
    config.CSTAR_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CSTAR_TABLE_DIR / name
    if not path.exists():
        path.write_text(text)
    return name


def describe(scenario) -> None:
    """Print every input of a scenario (preset name, Scenario, RunResult or dict) by section."""
    s = scenario_from(scenario)
    d = dataclasses.asdict(s)
    print(f"{d.pop('name')}  (level {d.pop('level')})")
    for key, value in d.items():
        if isinstance(value, dict) and key != "gases_ppb":
            fields = ", ".join(f"{k}={_short(v)}" for k, v in value.items())
            print(f"  {key}: {fields}")
        else:
            print(f"  {key} = {_short(value)}")


# --- input checks -------------------------------------------------------------------------------

def _sections() -> dict[str, type]:
    """Scenario fields that take a dict of their own inputs."""
    from tomas_jax.box import scenario as sc
    return {"environment": sc.Environment, "seed": sc.Seed, "photolysis": sc.Photolysis,
            "surrogate": sc.Surrogate, "processes": sc.Processes, "numerics": sc.Numerics}


def _allowed_inputs() -> dict[str, list[str] | None]:
    """Top-level input -> its nested inputs (None for plain values)."""
    from tomas_jax.box.scenario import Scenario
    sections = _sections()
    allowed = {}
    for f in dataclasses.fields(Scenario):
        if f.name in sections:
            allowed[f.name] = [g.name for g in dataclasses.fields(sections[f.name])]
            if f.name == "seed":
                allowed[f.name] += list(config.SEED_MODE_KEYS)
        else:
            allowed[f.name] = None
    return allowed


def _check_inputs(overrides: dict) -> None:
    """ValueError naming an unknown input, with close matches and the list of valid inputs."""
    allowed = _allowed_inputs()
    spellings = {}                                  # bare name -> how to write it
    for key, nested in allowed.items():
        spellings[key] = f"{key}=..."
        for sub in nested or ():
            spellings.setdefault(sub, f"{key}={{'{sub}': ...}}")
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"unknown input {key!r}.{_suggest(key, spellings)} Valid inputs: "
                             f"{_list_inputs(allowed)}")
        nested = allowed[key]
        if nested is None or not isinstance(value, dict):
            continue
        for sub in value:
            if sub not in nested:
                options = {n: f"{key}={{'{n}': ...}}" for n in nested}
                raise ValueError(f"unknown input {key}={{'{sub}': ...}}.{_suggest(sub, options)} "
                                 f"{key} accepts: {', '.join(nested)}")


def _suggest(name: str, spellings: dict[str, str]) -> str:
    close = difflib.get_close_matches(name, list(spellings), n=3, cutoff=0.6)
    return f" Did you mean {' or '.join(spellings[c] for c in close)}?" if close else ""


def _list_inputs(allowed) -> str:
    return ", ".join(key if nested is None else f"{key}={{{', '.join(nested)}}}"
                     for key, nested in allowed.items())


def _expand_seed(current, changes: dict) -> dict:
    """seed={"number_cm3": ...} etc. -> a full Seed dict with the first mode changed."""
    mode_changes = {k: v for k, v in changes.items() if k in config.SEED_MODE_KEYS}
    if not mode_changes:
        return changes
    if "modes" in changes:
        raise ValueError("give either seed={'modes': [...]} or number_cm3/gmd_nm/gsd, not both")
    modes = [dataclasses.asdict(m) for m in current.modes]
    modes[0] = {**modes[0], **mode_changes}
    rest = {k: v for k, v in changes.items() if k not in config.SEED_MODE_KEYS}
    return {**rest, "modes": modes}


@contextlib.contextmanager
def _cstar_table_visible(table: str | None):
    """Let the model find a table written by write_cstar_table.

    The pinned tomas-jax only reads tables from its own mechanisms folder, so the run
    temporarily points that folder at CSTAR_TABLE_DIR.
    """
    from tomas_jax.box import driver
    ours = table and (config.CSTAR_TABLE_DIR / table).exists()
    if ours and not (driver.MECHANISM_DIR / table).exists():
        with mock.patch.object(driver, "MECHANISM_DIR", config.CSTAR_TABLE_DIR):
            yield
    else:
        yield


# --- printing -------------------------------------------------------------------------------------

def _short(value):
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_short(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return repr(value) if isinstance(value, str) else str(value)


def _source_of(dist: str) -> str:
    """'<commit7> (git)' for git installs, the path for local ones, else the version."""
    try:
        d = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        return "not installed"
    direct = d.read_text("direct_url.json")
    if direct:
        info = json.loads(direct)
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return f"{commit[:7]} (git, version {d.version})"
        return f"local {info.get('url', '')} (version {d.version})"
    return d.version


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


class _Progress:
    """Progress bar that updates in place in Jupyter, Colab and VS Code (a text line elsewhere)."""

    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.fraction = 0.0
        self.handle = None
        if _in_notebook():
            from IPython.display import HTML, display
            self._html = HTML
            self.handle = display(HTML(self._render(0.0, "starting (first run compiles)")),
                                  display_id=True)

    def update(self, fraction: float) -> None:
        now = time.perf_counter()
        if fraction < 1.0 and now - self.last < config.PROGRESS_MIN_INTERVAL_S:
            return
        self.last = now
        self._show(fraction, f"{100 * fraction:.0f}% · {now - self.t0:.0f} s")

    def finish(self, status: str) -> None:
        self._show(1.0 if status.startswith("done") else self.fraction, status, final=True)

    def _show(self, fraction: float, status: str, final: bool = False) -> None:
        self.fraction = fraction
        if self.handle is not None:
            self.handle.update(self._html(self._render(fraction, status)))
        else:
            print(f"\r{self.label}: {status}    ", end="\n" if final else "", flush=True)

    def _render(self, fraction: float, status: str) -> str:
        return (f'<div style="font-family:monospace">{self.label} '
                f'<progress value="{100 * fraction:.0f}" max="100" style="width:240px; '
                f'vertical-align:middle"></progress> {status}</div>')
