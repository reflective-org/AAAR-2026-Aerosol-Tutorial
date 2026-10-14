"""Interactive Plotly figures of box-model results.

Every function takes a RunResult (or a list of them for comparisons) and returns a
``plotly.graph_objects.Figure``; show it with ``fig.show()`` or by ending a cell with it.
Time is hours since the run start; shaded bands mark night (solar zenith angle > 90 deg).
``style(fig, title)`` gives your own figures the same look.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .results import as_result, has_nitrogen, night_intervals, series

_LOG_DIAMETER = dict(type="log", tickvals=[m * 10.0 ** e for e in range(0, 5) for m in (1, 2, 5)],
                     ticktext=[f"{m * 10 ** e:g}" for e in range(0, 5) for m in (1, 2, 5)])
_DIST = {
    "number": ("dndlog10dp_cm3", "dN/dlog₁₀Dp (cm⁻³)", "number"),
    "mass": ("dmdlog10dp_ugm3", "dM/dlog₁₀Dp (µg m⁻³)", "dry mass"),
}


# --- size distributions -----------------------------------------------------------------------

def size_distribution(result, quantity: str = "number", log_y: bool = False,
                      snapshots_every_h: float | None = 1.0) -> go.Figure:
    """Number or dry-mass size distribution with a time slider and play button.

    Grey: initial distribution; faint lines: snapshots every ``snapshots_every_h`` hours;
    bold: the time selected on the slider.
    """
    r = as_result(result)
    t, d, z = _dist(r, quantity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=z[0], name="initial", line=dict(color=config.INITIAL_COLOR,
                                                                      dash="dot", width=2)))
    if snapshots_every_h:
        for k, i in enumerate(_every(t, snapshots_every_h)):
            fig.add_trace(go.Scatter(x=d, y=z[i], name=f"every {snapshots_every_h:g} h",
                                     legendgroup="snapshots", showlegend=k == 0, hoverinfo="skip",
                                     line=dict(color=config.SERIES_COLORS[0], width=1), opacity=0.25))
    current = len(fig.data)
    fig.add_trace(_dist_trace(d, z[-1], f"t = {_hours(t[-1])} h", config.SERIES_COLORS[0]))
    frames = [go.Frame(name=str(i), traces=[current],
                       data=[_dist_trace(d, z[i], f"t = {_hours(t[i])} h", config.SERIES_COLORS[0])])
              for i in range(t.size)]
    _add_time_slider(fig, t, frames)
    label = _DIST[quantity][1]
    fig.update_xaxes(title=config.DIAMETER_AXIS_TITLE, range=_diameter_range(r, z), **_LOG_DIAMETER)
    fig.update_yaxes(title=label, **_fixed_y(z, log_y))
    return style(fig, f"{_name(r)}: {_DIST[quantity][2]} size distribution")


def compare_size_distributions(results, labels=None, quantity: str = "number",
                               time_h: float | None = None, log_y: bool = False) -> go.Figure:
    """Size distributions of several runs at one time (default: the end), and where they started.

    Dotted: the initial distribution, one grey curve if all runs start the same, else one
    per run in its colour.
    """
    runs, labels = _runs(results, labels)
    picked = []                                    # (time, diameters, initial, selected)
    for r in runs:
        t, d, z = _dist(r, quantity)
        i = t.size - 1 if time_h is None else int(np.argmin(np.abs(t - time_h)))
        picked.append((float(t[i]), d, z[0], z[i]))
    same_start = all(p[2].shape == picked[0][2].shape and np.allclose(p[2], picked[0][2])
                     for p in picked)
    same_time = all(abs(p[0] - picked[0][0]) < 1e-9 for p in picked)
    fig = go.Figure()
    if same_start:
        fig.add_trace(go.Scatter(x=picked[0][1], y=picked[0][2], name="initial",
                                 line=dict(color=config.INITIAL_COLOR, dash="dot", width=2)))
    for j, ((t_i, d, z0, z_i), label) in enumerate(zip(picked, labels)):
        if not same_start:
            fig.add_trace(go.Scatter(x=d, y=z0, name=f"{label}: initial", legendgroup=label,
                                     line=dict(color=_color(j), dash="dot", width=1.5)))
        name = label if same_time else f"{label} (t = {_hours(t_i)} h)"
        fig.add_trace(_dist_trace(d, z_i, name, _color(j)).update(legendgroup=label))
    all_z = np.array([p[2] for p in picked] + [p[3] for p in picked])
    fig.update_xaxes(title=config.DIAMETER_AXIS_TITLE, range=_diameter_range(runs[0], all_z),
                     **_LOG_DIAMETER)
    fig.update_yaxes(title=_DIST[quantity][1], **_fixed_y(all_z, log_y))
    if same_time:
        when = f"t = {_hours(picked[0][0])} h"
    else:
        when = "the end of each run" if time_h is None else f"t ≈ {time_h:g} h"
    return style(fig, f"{_DIST[quantity][2].capitalize()} size distributions at {when}",
                 legend_right=len(runs) > 3 or not same_start)


def banana(result, quantity: str = "number", decades: float = config.HEATMAP_DECADES) -> go.Figure:
    """Size distribution over time as a heatmap (log colour scale): the "banana plot"."""
    r = as_result(result)
    t, _, z = _dist(r, quantity)
    edges = np.asarray(r.diameter_edges_nm, dtype=float)
    top = np.nanmax(z)
    if not top > 0:
        return _message_figure("The size distribution is empty.")
    floor = top * 10.0 ** -decades
    with np.errstate(divide="ignore", invalid="ignore"):
        logz = np.where(z >= floor, np.log10(z), np.nan)
    ticks = np.arange(np.ceil(np.log10(floor)), np.floor(np.log10(top)) + 1)
    unit = "cm⁻³" if quantity == "number" else "µg m⁻³"
    fig = go.Figure(go.Heatmap(
        x=t, y=edges, z=logz.T, zmin=np.log10(floor), zmax=np.log10(top),
        colorscale=config.HEATMAP_COLORSCALE, customdata=z.T,
        colorbar=dict(title=dict(text=f"{_DIST[quantity][1].split(' (')[0]}<br>({unit})"),
                      tickvals=ticks, ticktext=[f"10<sup>{int(k)}</sup>" for k in ticks]),
        hovertemplate=(f"t = %{{x:.2f}} h<br>Dp = %{{y:.3g}} nm<br>value = %{{customdata:.3g}} {unit}"
                       "<extra></extra>")))
    fig.update_xaxes(title=config.TIME_AXIS_TITLE)
    fig.update_yaxes(title=config.DIAMETER_AXIS_TITLE, range=_diameter_range(r, z, floor / top),
                     **_LOG_DIAMETER)
    return style(fig, f"{_name(r)}: {_DIST[quantity][2]} distribution over time")


# --- time series ------------------------------------------------------------------------------

def time_series(results, keys, labels=None, log_y: bool = False, title: str | None = None) -> go.Figure:
    """One panel per series name (see results.series), one line per run; shared time axis.

    With ``log_y=True``, zeros (e.g. radicals at t = 0) are left out instead of drawn at the
    bottom of the axis. N2O5 is shown in ppt (config.DISPLAY_SCALE).
    """
    runs, labels = _runs(results, labels)
    keys = [keys] if isinstance(keys, str) else list(keys)
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    names = []
    for row, key in enumerate(keys, start=1):
        name, unit = config.SERIES_LABELS.get(key, (key, ""))
        names.append(name)
        panel = []
        for j, (r, label) in enumerate(zip(runs, labels)):
            y = series(r, key) * config.DISPLAY_SCALE.get(key, 1.0)
            if log_y:
                y = _positive(y)
            panel.append(y)
            color = _color(j) if len(runs) > 1 else _color(row - 1)
            fig.add_trace(go.Scatter(
                x=r.time_h, y=y, name=label if len(runs) > 1 else name,
                legendgroup=label, showlegend=len(runs) > 1 and row == 1, line=dict(color=color),
                hovertemplate=f"{label}<br>t = %{{x:.2f}} h<br>{name} = %{{y:.3g}} {unit}<extra></extra>"),
                row=row, col=1)
        axis = _log_axis(*panel) if log_y else {}
        fig.update_yaxes(title=f"{name}<br>({unit})" if unit else name, row=row, col=1, **axis)
    fig.update_xaxes(title=config.TIME_AXIS_TITLE, row=len(keys), col=1)
    _shade_nights(fig, runs[0], len(runs) > 1)
    height = max(config.PLOT_HEIGHT, config.PLOT_HEIGHT_PER_PANEL * len(keys))
    if title is None:
        title = _name(runs[0]) if len(runs) == 1 else _join(names)
    return style(fig, title, height)


def totals(results, labels=None, keys=config.DEFAULT_TOTALS, log_y: bool = False) -> go.Figure:
    """Particle number, SOA, sulfate, nitrate and dry mass over time (one or several runs)."""
    runs, labels = _runs(results, labels)
    title = f"{_name(runs[0])}: totals" if len(runs) == 1 else "Totals"
    return time_series(runs, keys, labels, log_y=log_y, title=title)


def composition(result, fraction: bool = False) -> go.Figure:
    """Dry particle mass by component over time, stacked (``fraction=True``: percent of dry mass)."""
    r = as_result(result)
    fig = make_subplots(rows=1, cols=1)
    for key, label, color in config.COMPOSITION_LAYERS:
        fig.add_trace(go.Scatter(
            x=r.time_h, y=series(r, key), name=label, stackgroup="mass", mode="lines",
            line=dict(width=0.5, color=color), fillcolor=color,
            groupnorm="percent" if fraction else None,
            hovertemplate=f"{label}: %{{y:.3g}}{' %' if fraction else ' µg m⁻³'}<extra></extra>"))
    fig.update_xaxes(title=config.TIME_AXIS_TITLE)
    fig.update_yaxes(title="Share of dry mass (%)" if fraction else "Dry particle mass (µg m⁻³)",
                     rangemode="tozero")
    _shade_nights(fig, r)
    fig.update_layout(hovermode="x unified")
    return style(fig, f"{_name(r)}: particle composition")


def gases(result, species=None, log_y: bool = True) -> go.Figure:
    """Gas mixing ratios (ppb). Default: every reported gas, the level's main ones visible."""
    r = as_result(result)
    default = config.DEFAULT_GASES.get(r.scenario.get("level"), ())
    return _species_plot(r, "gases_ppb", species, default, log_y, "Mixing ratio", "ppb", "gases")


def radicals(result, species=config.DEFAULT_RADICALS, log_y: bool = True) -> go.Figure:
    """OH, HO2 and NO3 concentrations (molecules per cm3)."""
    r = as_result(result)
    return _species_plot(r, "radicals_cm3", species, species, log_y, "Concentration", "molec cm⁻³",
                         "radicals")


def photolysis(result, j=config.DEFAULT_J, show_sza: bool = True, log_y: bool = True) -> go.Figure:
    """Selected photolysis frequencies J (1/s) and, below, the solar zenith angle."""
    r = as_result(result)
    if not r.photolysis:
        return _message_figure("No photolysis in this run (Level 1 has no gas chemistry).")
    j = [j] if isinstance(j, str) else list(j)
    rows = 2 if show_sza else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.65, 0.35] if show_sza else None)
    values = []
    for k, name in enumerate(j):
        label = config.SERIES_LABELS.get(name, (name, ""))[0]
        y = _positive(series(r, name)) if log_y else series(r, name)
        values.append(y)
        fig.add_trace(go.Scatter(x=r.time_h, y=y, name=label, line=dict(color=_color(k)),
                                 hovertemplate=f"{label} = %{{y:.3g}} s⁻¹<extra></extra>"), row=1, col=1)
    fig.update_yaxes(title="J (s⁻¹)", row=1, col=1, **(_log_axis(*values) if log_y else {}))
    if show_sza:
        fig.add_trace(go.Scatter(x=r.time_h, y=series(r, "sza_deg"), name="solar zenith angle",
                                 line=dict(color="#555555"),
                                 hovertemplate="SZA = %{y:.1f}°<extra></extra>"), row=2, col=1)
        fig.add_hline(y=config.NIGHT_SZA_DEG, line=dict(color="#999999", dash="dash", width=1),
                      row=2, col=1)
        fig.update_yaxes(title="SZA (°)", range=[0, 180], row=2, col=1)
    fig.update_xaxes(title=config.TIME_AXIS_TITLE, row=rows, col=1)
    _shade_nights(fig, r)
    mode = r.scenario.get("photolysis", {}).get("mode", "")
    return style(fig, f"{_name(r)}: sunlight (photolysis mode “{mode}”)",
                  config.PLOT_HEIGHT + (120 if show_sza else 0))


def sunlight(days, j: str = "J_NO2") -> go.Figure:
    """J and solar zenith angle against local clock time for results of sunlight_over_day()."""
    days = [days] if isinstance(days, dict) else list(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    label = config.SERIES_LABELS.get(j, (j, ""))[0]
    for n, day in enumerate(days):
        x = day["local_start_h"] + np.asarray(day["time_h"])
        line = dict(color=_color(n))
        fig.add_trace(go.Scatter(x=x, y=day[j], name=day["label"], legendgroup=day["label"], line=line,
                                 hovertemplate=f"{label} = %{{y:.3g}} s⁻¹<extra>{day['label']}</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=day["sza_deg"], name=day["label"], legendgroup=day["label"],
                                 showlegend=False, line=line,
                                 hovertemplate=f"SZA = %{{y:.1f}}°<extra>{day['label']}</extra>"),
                      row=2, col=1)
    fig.add_hline(y=config.NIGHT_SZA_DEG, line=dict(color="#999999", dash="dash", width=1), row=2, col=1)
    first = min(day["local_start_h"] for day in days)
    last = max(day["local_start_h"] + float(np.max(day["time_h"])) for day in days)
    ticks = list(range(int(first), int(last) + 1, 3))
    fig.update_xaxes(title="Local clock time", tickvals=ticks, ticktext=[f"{h % 24:02d}:00" for h in ticks],
                     row=2, col=1)
    fig.update_yaxes(title=f"{label} (s⁻¹)", row=1, col=1)
    fig.update_yaxes(title="SZA (°)", range=[0, 180], row=2, col=1)
    return style(fig, "Sunlight over the day (TUV-x, clear sky); SZA above the dashed line: night",
                 config.PLOT_HEIGHT + 140)


# --- organics -----------------------------------------------------------------------------------

def volatility(result) -> go.Figure:
    """Organic mass per c* class, particle stacked under gas, with a time slider.

    The dashed line is C_OA = SOA + POA; classes left of it are mostly in the particles
    (particle fraction 1 / (1 + c*/C_OA) > 1/2).
    """
    r = as_result(result)
    vol = r.volatility
    if not len(vol.get("log10_cstar", [])):
        return _message_figure("No volatility classes in this run.")
    x = np.asarray(vol["log10_cstar"], dtype=float)
    part = np.asarray(vol["particle_ugm3"], dtype=float)
    gas = np.asarray(vol["gas_ugm3"], dtype=float)
    t = np.asarray(r.time_h, dtype=float)
    coa = series(r, "SOA_ugm3") + series(r, "POA_ugm3")
    y_max = max(float(np.nanmax(part + gas)), 1e-12) * 1.1
    log_coa = np.log10(coa, where=coa > 0, out=np.full_like(coa, np.nan))
    lo = min(x.min(), np.nanmin(log_coa, initial=x.min()), -3.0) - 0.5
    hi = max(x.max(), np.nanmax(log_coa, initial=x.max()), 4.0) + 0.5

    def traces(i):
        hover = "c* = 10<sup>%{x:.1f}</sup> µg m⁻³<br>%{y:.3g} µg m⁻³<extra>%{fullData.name}</extra>"
        coa_x = [log_coa[i]] * 2 if np.isfinite(log_coa[i]) else [None, None]
        return [go.Bar(x=x, y=part[i], name="particle phase", marker_color=config.PARTICLE_COLOR,
                       width=0.4, hovertemplate=hover),
                go.Bar(x=x, y=gas[i], name="gas phase", marker_color=config.GAS_COLOR, width=0.4,
                       hovertemplate=hover),
                go.Scatter(x=coa_x, y=[0, y_max], mode="lines", name=f"C_OA = {coa[i]:.3g} µg m⁻³",
                           line=dict(color="#444444", dash="dash"), hoverinfo="skip")]

    last = t.size - 1
    fig = go.Figure(traces(last))
    frames = [go.Frame(name=str(i), data=traces(i), traces=[0, 1, 2]) for i in range(t.size)]
    _add_time_slider(fig, t, frames)
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    fig.update_xaxes(title="Saturation concentration c* at 298 K (µg m⁻³)", range=[lo, hi],
                     tickvals=ticks, ticktext=[f"10<sup>{int(k)}</sup>" for k in ticks])
    fig.update_yaxes(title="Organic mass (µg m⁻³)", range=[0, y_max])
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    return style(fig, f"{_name(r)}: organic mass by volatility class")


def top_contributors(result, n: int = 15) -> go.Figure:
    """Organic products with the most particle-phase mass at the end of the run."""
    r = as_result(result)
    rows = sorted(r.top_contributors, key=lambda c: c["particle_ugm3"], reverse=True)[:n]
    if not rows:
        return _message_figure("No species-resolved SOA in this run (Level 1 uses one surrogate).")
    rows = rows[::-1]                                      # largest on top
    has_n = [has_nitrogen(c["formula"]) for c in rows]
    total = sum(c["particle_ugm3"] for c in rows)
    share_n = sum(c["particle_ugm3"] for c, flag in zip(rows, has_n) if flag) / total
    fig = go.Figure()
    for flag, label, color in ((True, "contains N (nitrate, PAN)", config.SERIES_COLORS[1]),
                               (False, "no N", config.PARTICLE_COLOR)):
        sel = [c for c, f in zip(rows, has_n) if f == flag]
        fig.add_trace(go.Bar(
            y=[f"{c['species']} · {c['formula']}" for c in sel], x=[c["particle_ugm3"] for c in sel],
            orientation="h", name=label, marker_color=color,
            customdata=[[c["formula"], c["log10_cstar"]] for c in sel],
            hovertemplate=("%{y}<br>particle: %{x:.3g} µg m⁻³<br>log₁₀ c* = %{customdata[1]:.2f}"
                           "<extra></extra>")))
    order = [f"{c['species']} · {c['formula']}" for c in rows]
    fig.update_yaxes(categoryorder="array", categoryarray=order)
    fig.update_xaxes(title="Particle-phase mass at end of run (µg m⁻³)")
    height = max(config.PLOT_HEIGHT, 28 * len(rows) + 160)
    return style(fig, f"{_name(r)}: top {len(rows)} SOA species "
                       f"(N-containing: {100 * share_n:.0f}% of their mass)", height)


# --- helpers ------------------------------------------------------------------------------------

def _species_plot(r, family, species, visible, log_y, y_title, unit, what) -> go.Figure:
    values = getattr(r, family)
    if not values:
        return _message_figure(f"No {what} in this run (Level 1 has no gas chemistry).")
    show_all = species is None
    names = list(values) if show_all else ([species] if isinstance(species, str) else list(species))
    fig = make_subplots(rows=1, cols=1)
    order = list(values)
    shown = []
    for name in names:
        label = config.SERIES_LABELS.get(name, (name, ""))[0]
        y = _positive(series(r, name)) if log_y else series(r, name)
        is_visible = not show_all or name in visible
        if is_visible:
            shown.append(y)
        fig.add_trace(go.Scatter(
            x=r.time_h, y=y, name=label,
            line=dict(color=_color(order.index(name) if name in order else len(order))),
            visible=True if is_visible else "legendonly",
            hovertemplate=f"{label} = %{{y:.3g}} {unit}<extra></extra>"))
    fig.update_xaxes(title=config.TIME_AXIS_TITLE)
    fig.update_yaxes(title=f"{y_title} ({unit})", **(_log_axis(*shown) if log_y else {}))
    _shade_nights(fig, r)
    title = f"{_name(r)}: {what}" + (" (click a legend entry to show or hide it)" if show_all else "")
    return style(fig, title, legend_right=True)


def _dist(r, quantity):
    if quantity not in _DIST:
        raise ValueError(f"quantity must be one of {list(_DIST)}")
    z = np.asarray(getattr(r, _DIST[quantity][0]), dtype=float)
    return np.asarray(r.time_h, dtype=float), np.asarray(r.diameter_nm, dtype=float), z


def _dist_trace(d, y, name, color):
    return go.Scatter(x=d, y=y, name=name, line=dict(color=color, width=3),
                      hovertemplate="Dp = %{x:.3g} nm<br>%{y:.3g}<extra></extra>")


def _diameter_range(r, z, rel_floor=1e-3):
    """Log10 x range covering the bins that ever hold more than rel_floor of the peak."""
    edges = np.asarray(r.diameter_edges_nm, dtype=float)
    peak = np.nanmax(z, axis=tuple(range(z.ndim - 1))) if z.ndim > 1 else z
    used = np.flatnonzero(peak > rel_floor * np.nanmax(peak)) if np.nanmax(peak) > 0 else []
    if len(used) == 0:
        return [np.log10(edges[0]), np.log10(edges[-1])]
    lo, hi = edges[max(used[0] - 1, 0)], edges[min(used[-1] + 2, edges.size - 1)]
    return [np.log10(lo), np.log10(hi)]


def _fixed_y(z, log_y):
    top = float(np.nanmax(z)) if np.size(z) else 0.0
    if not top > 0:
        return {}
    if log_y:
        return dict(type="log", range=[np.log10(top) - 4, np.log10(top) + 0.1], exponentformat="power")
    return dict(range=[0, 1.05 * top])


def _every(t, step_h):
    """Indices of the first output at or after each multiple of step_h (excluding t = 0)."""
    targets = np.arange(step_h, t[-1] + 1e-9, step_h)
    return sorted({int(i) for i in np.searchsorted(t, targets - 1e-9) if i < t.size})


def _add_time_slider(fig, t, frames):
    fig.frames = frames
    steps = [dict(method="animate", label=_hours(ti),
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for i, ti in enumerate(t)]
    fig.update_layout(
        sliders=[dict(active=len(steps) - 1, steps=steps, x=0.1, len=0.9, xanchor="left", y=0,
                      yanchor="top", pad=dict(t=50, b=10), ticklen=2,
                      currentvalue=dict(prefix="t = ", suffix=" h", xanchor="right",
                                        font=dict(size=14, color="#333")),
                      font=dict(color="rgba(0,0,0,0)"))],            # hide the crowded step labels
        updatemenus=[dict(type="buttons", direction="left", x=0.1, y=0, xanchor="right", yanchor="top",
                          pad=dict(t=60, r=10), showactive=False, buttons=[
                              dict(label="▶", method="animate", args=[None, dict(
                                  frame=dict(duration=config.SLIDER_FRAME_MS, redraw=True),
                                  transition=dict(duration=0), fromcurrent=False)]),
                              dict(label="❚❚", method="animate", args=[[None], dict(
                                  frame=dict(duration=0, redraw=False), mode="immediate",
                                  transition=dict(duration=0))])])],
        margin=dict(b=140))


def _shade_nights(fig, r, first_run_only=False):
    intervals = night_intervals(r)
    if not intervals:
        return
    for t0, t1 in intervals:
        fig.add_vrect(x0=t0, x1=t1, fillcolor=config.NIGHT_COLOR, line_width=0, layer="below",
                      row="all", col=1)
    label = "night (SZA > 90°)" + (", first run" if first_run_only else "")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, hoverinfo="skip",
                             marker=dict(symbol="square", size=14, color=config.NIGHT_COLOR,
                                         line=dict(color="#999999", width=1))), row=1, col=1)


def _runs(results, labels):
    if not isinstance(results, (list, tuple)):
        results = [results]
    runs = [as_result(r) for r in results]
    if labels is None:
        labels = [_name(r) for r in runs]
        if len(set(labels)) < len(labels):
            labels = [f"run {i + 1}" for i in range(len(runs))]
    if len(labels) != len(runs):
        raise ValueError(f"got {len(runs)} results but {len(labels)} labels")
    return runs, list(labels)


def _name(r):
    return r.scenario.get("name", "run")


def _color(i):
    return config.SERIES_COLORS[i % len(config.SERIES_COLORS)]


def _hours(t):
    return f"{t:.2f}".rstrip("0").rstrip(".")


def _positive(y):
    """Values <= 0 become nan, which a log axis leaves out as a gap."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def _log_axis(*arrays) -> dict:
    """Log axis with readable tick labels for the range of the positive values in ``arrays``.

    Narrow ranges get evenly spaced ticks, up to ~3 decades get 1, 2 and 5 per decade, wider
    ranges powers of ten only. All labels on one axis use the same style.
    """
    v = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays]) if arrays else np.array([])
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return dict(type="log")
    lo, hi = float(v.min()), float(v.max())
    span = np.log10(hi / lo)
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    if span < config.LOG_AXIS_LINEAR_BELOW_DECADES:
        step = _nice_step((hi - lo) / 4) if hi > lo else 10.0 ** np.floor(np.log10(lo))
        ticks = [t for t in np.arange(np.floor(lo / step), np.ceil(hi / step) + 1) * step if t > 0]
    elif span <= 1.2:
        ticks = [m * 10.0 ** e for e in decades for m in (1, 2, 3, 5)]
    elif span <= config.LOG_AXIS_DETAIL_DECADES:
        ticks = [m * 10.0 ** e for e in decades for m in (1, 2, 5)]
    else:
        ticks = [10.0 ** e for e in decades[::max(1, int(np.ceil(decades.size / 8)))]]
    visible = [t for t in ticks if lo / 3 <= t <= hi * 3]
    scientific = any(t < 1e-3 or t >= 1e5 for t in visible)
    return dict(type="log", tickvals=ticks, ticktext=[_tick_label(t, scientific) for t in ticks])


def _nice_step(x):
    exponent = np.floor(np.log10(x))
    fraction = x / 10.0 ** exponent
    return (1 if fraction < 1.5 else 2 if fraction < 3.5 else 5 if fraction < 7.5 else 10) * 10.0 ** exponent


def _tick_label(value, scientific=False):
    """'2000' or '0.005'; with scientific=True '2×10⁵' style (1 and 10 stay plain)."""
    if not scientific:
        return f"{value:.6g}"
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = float(f"{value / 10.0 ** exponent:.3g}")
    if mantissa == 1 and exponent in (0, 1):
        return f"{value:g}"
    return f"10<sup>{exponent}</sup>" if mantissa == 1 else f"{mantissa:g}×10<sup>{exponent}</sup>"


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _message_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=15))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, "", 240)


def style(fig, title, height=None, legend_right=False) -> go.Figure:
    """Give any Plotly figure the look of these plots: template, font, title, legend, margins.

    The legend goes above the plot, or to its right with ``legend_right=True``.
    """
    legend = (dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02) if legend_right
              else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_layout(
        template=config.PLOT_TEMPLATE, height=height or config.PLOT_HEIGHT,
        font=dict(size=config.PLOT_FONT_SIZE), title=dict(text=title, x=0.01, xanchor="left"),
        legend=legend, margin=dict(l=80, r=30, t=90, b=fig.layout.margin.b or 60))
    return fig
