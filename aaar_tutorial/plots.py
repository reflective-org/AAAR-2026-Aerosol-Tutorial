"""Interactive Plotly figures of box-model results.

Every function takes a RunResult (or a list of them for comparisons) and returns a
``plotly.graph_objects.Figure``; show it with ``fig.show()`` or by ending a cell with it
(inside a widget, e.g. a run panel's plot area, use ``show_in_widget(fig)``).
``partitioning()`` needs no run: it draws the equilibrium partitioning equation.
Time is hours since the run start; shaded bands mark night (solar zenith angle > 90 deg).
``style(fig, title)`` gives your own figures the same look.

The look is the chart style of the tutorial web app: the subtitle names what is plotted and
its units, the first tick carries the unit of the x axis ("0 h", "10 nm"), lines are told apart
by colour, dash and a label at the line end, and two runs are drawn as A (Charcoal, solid) and
B (Blue, dashed). The legends stay: click an entry to hide or show a line (and its label).
"""
from __future__ import annotations

import html
import re
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config, theme  # noqa: F401  (theme registers the template)
from .results import as_result, has_nitrogen, night_intervals, series

_DIST = {   # quantity -> (RunResult attribute, y quantity and unit, title word)
    "number": ("dndlog10dp_cm3", "dN/dlog₁₀Dp (cm⁻³)", "number"),
    "mass": ("dmdlog10dp_ugm3", "dM/dlog₁₀Dp (µg/m³)", "dry mass"),
}
_DIAMETER_SUBTITLE = "dry diameter Dp (nm)"
_TIME_SUBTITLE = "time since start (h)"
_INITIAL_LINE = dict(color=config.INITIAL_COLOR, dash="dot", width=1.6)
_NIGHT_SZA_LINE = dict(color=config.REFERENCE_LINE_COLOR, dash=config.REFERENCE_DASH, width=1.2)
_LABEL_GAP_PX = 14                 # labels at line ends: closest spacing
_END_LABEL_LEAD = "\u2002"         # an en space (about 6 px) between a line's end and its label
_LABEL_MAX_CHARS = 24              # longer run names stay in the legend only
_CHAR_PX = 6.6                     # average width of a label character (px, estimate)
_PLOT_WIDTH_PX = 850               # plot-area width used to place labels (px, estimate)
_MARGIN = dict(l=56, r=24, b=40)


# --- size distributions -----------------------------------------------------------------------

def size_distribution(result, quantity: str = "number", log_y: bool = False,
                      snapshots_every_h: float | None = 1.0) -> go.Figure:
    """Number or dry-mass size distribution with a time slider and play button.

    Grey dotted: initial distribution; faint lines: snapshots every ``snapshots_every_h`` hours;
    bold: the time selected on the slider.
    """
    r = as_result(result)
    t, d, z = _dist(r, quantity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=z[0], name="initial", mode="lines", line=_INITIAL_LINE))
    if snapshots_every_h:
        for k, i in enumerate(_every(t, snapshots_every_h)):
            fig.add_trace(go.Scatter(x=d, y=z[i], name=f"every {snapshots_every_h:g} h", mode="lines",
                                     legendgroup="snapshots", showlegend=k == 0, hoverinfo="skip",
                                     line=dict(color=config.SNAPSHOT_COLOR, width=1, dash="solid")))
    current = len(fig.data)
    bold = dict(mode="lines", line=dict(color=config.SELECTED_COLOR, width=config.PLOT_SELECTED_LINE_WIDTH,
                                        dash="solid"))
    fig.add_trace(_dist_trace(d, z[-1], f"{_hours(t[-1])} h", bold))
    frames = [([_dist_trace(d, z[i], f"{_hours(t[i])} h", bold)], [current]) for i in range(t.size)]
    _apply(fig, "xaxis", _diameter_axis(_diameter_range(r, z)))
    scales = _y_scales(fixed=z)
    _apply(fig, "yaxis", scales[int(log_y)])
    _add_scale_buttons(fig, {"yaxis": scales}, log_y)
    _add_time_slider(fig, t, frames)
    return style(fig, f"{_name(r)}: {_DIST[quantity][2]} size distribution",
                 subtitle=f"{_DIST[quantity][1]} vs {_DIAMETER_SUBTITLE}")


def compare_size_distributions(results, labels=None, quantity: str = "number",
                               time_h: float | None = None, log_y: bool = False) -> go.Figure:
    """Size distributions of several runs at one time (default: the end), and where they started.

    Two runs: A in Charcoal (solid), B in Blue (dashed), as in the web app; more runs: each its
    own colour and dash. Each curve is labelled at its peak. Dotted: the initial distribution,
    one grey curve if all runs start the same, else one per run in its colour.
    """
    runs, labels = _runs(results, labels)
    looks, names, tags = _run_looks(labels)
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
        fig.add_trace(go.Scatter(x=picked[0][1], y=picked[0][2], name="initial", mode="lines",
                                 line=_INITIAL_LINE))
    for j, ((t_i, d, z0, z_i), label) in enumerate(zip(picked, labels)):
        if not same_start:
            fig.add_trace(go.Scatter(x=d, y=z0, name=f"{label}: initial", legendgroup=label, mode="lines",
                                     line=dict(color=looks[j]["color"], dash="dot", width=1.4)))
        name = names[j] if same_time else f"{names[j]} (t = {_hours(t_i)} h)"
        fig.add_trace(_dist_trace(d, z_i, name, dict(mode="lines", line=looks[j])).update(legendgroup=label))
    # runs may use different size grids (e.g. 40 and 160 bins): union of their ranges
    ranges = [_diameter_range(r, np.array([p[2], p[3]])) for r, p in zip(runs, picked)]
    _apply(fig, "xaxis", _diameter_axis([min(lo for lo, _ in ranges), max(hi for _, hi in ranges)]))
    all_z = np.concatenate([np.ravel(p[k]) for p in picked for k in (2, 3)])
    scales = _y_scales(fixed=all_z)
    _apply(fig, "yaxis", scales[int(log_y)])
    ys = {}
    if all(tags):                                  # a label over each run's peak
        room = 34 / _panel_px(config.PLOT_HEIGHT, 1)   # space above the highest peak for its label
        for scale in scales:
            lo, hi = scale["range"]
            scale["range"] = [lo, hi + room * (hi - lo)]
        _apply(fig, "yaxis", scales[int(log_y)])
        peaks = [(int(np.nanargmax(p[3])), p) for p in picked]
        x = [np.log10(p[1][k]) for k, p in peaks]
        y = [float(p[3][k]) for k, p in peaks]
        (lin, log), where = _spread_2d(x, y, [(scales[0]["range"], False), (scales[1]["range"], True)], tags)
        for j, (label, tag) in enumerate(zip(labels, tags)):
            ys[len(fig.data)] = ([lin[j]], [log[j]])
            text = {"top left": f"{tag}\u2009", "top right": f"\u2009{tag}"}.get(where[j], tag)
            fig.add_trace(_label_trace(10 ** x[j], (log if log_y else lin)[j], text, looks[j]["color"], label,
                                       position=where[j], bold=len(runs) == 2))
    _add_scale_buttons(fig, {"yaxis": scales}, log_y, ys=ys or None)
    if same_time:
        when = f"t = {_hours(picked[0][0])} h"
    else:
        when = "the end of each run" if time_h is None else f"t ≈ {time_h:g} h"
    if len(runs) == 2:                             # caption under the plot, as in the app
        modes = [p[1][int(np.nanargmax(p[3]))] for p in picked]
        _caption(fig, f"Mode {modes[0]:.0f} nm (A) vs {modes[1]:.0f} nm (B)")
    return style(fig, f"{_DIST[quantity][2].capitalize()} size distributions at {when}",
                 subtitle=f"{_DIST[quantity][1]} vs {_DIAMETER_SUBTITLE}")


def banana(result, quantity: str = "number", decades: float = config.HEATMAP_DECADES,
           log_color: bool = True) -> go.Figure:
    """Size distribution over time as a heatmap: the "banana plot".

    White (empty) to dark blue (most particles). The buttons switch between a log colour scale
    (``decades`` below the maximum; anything lower is drawn white) and a linear one. The small
    colour bar at the top right shows the range.
    """
    r = as_result(result)
    t, _, z = _dist(r, quantity)
    edges = np.asarray(r.diameter_edges_nm, dtype=float)
    top = np.nanmax(z)
    if not top > 0:
        return _message_figure("The size distribution is empty.")
    floor = _round_down_1(top * 10.0 ** -decades)    # a round value at the low end of the bar
    zero_filled = np.nan_to_num(z, nan=0.0)
    unit = "cm⁻³" if quantity == "number" else "µg/m³"
    lin = dict(z=zero_filled.T, zmin=0.0, zmax=top)
    log = dict(z=np.log10(np.maximum(zero_filled, floor)).T,       # empty and low cells: white
               zmin=np.log10(floor), zmax=np.log10(top))
    ends = {False: ("0", f"{_tick_label(_round2(top))} {unit}"),
            True: (_tick_label(floor, log=True), f"{_tick_label(_round2(top))} {unit}")}
    first = log if log_color else lin
    length, thick = config.HEATMAP_COLORBAR_PX
    room = 84                                     # px right of the bar for its top label
    # The bar sits 13 px above the plot, which starts 88 px from the top (style(): title and
    # subtitle, no legend). In container units: in paper units Plotly would move the title up.
    above, top = 13, 88
    fig = go.Figure(go.Heatmap(
        x=t, y=edges, z=first["z"], zmin=first["zmin"], zmax=first["zmax"],
        colorscale=config.HEATMAP_COLORSCALE, customdata=z.T,
        colorbar=dict(orientation="h", lenmode="pixels", len=length + 2 * room, thickness=thick,
                      x=1, xanchor="right", xpad=room, y=1 - (top - above) / config.PLOT_HEIGHT,
                      yref="container", yanchor="middle", ypad=0,
                      showticklabels=False, ticks="", outlinewidth=0),
        hovertemplate=(f"t = %{{x:.2f}} h<br>Dp = %{{y:.3g}} nm<br>value = %{{customdata:.3g}} {unit}"
                       "<extra></extra>")))
    note = dict(xref="paper", yref="paper", y=1, yanchor="middle", yshift=above,
                showarrow=False, font=dict(size=config.PLOT_NOTE_FONT_SIZE, color=config.TEXT_2))
    fig.add_annotation(x=1, xshift=-(room + length + 4), xanchor="right", text=ends[log_color][0], **note)
    fig.add_annotation(x=1, xshift=-(room - 4), xanchor="left", text=ends[log_color][1], **note)
    _apply(fig, "xaxis", _time_axis(_time_range(t)))
    _apply(fig, "yaxis", _diameter_axis(_diameter_range(r, z, floor / top)))
    colour = {key: (lin[key], log[key]) for key in ("z", "zmin", "zmax")}
    notes = {f"annotations[{k}].text": (ends[False][k], ends[True][k]) for k in (0, 1)}
    _add_scale_buttons(fig, {}, log_color, labels=tuple(f"{s} colour" for s in config.SCALE_BUTTON_LABELS),
                       colour=colour, layout=notes)
    return style(fig, f"{_name(r)}: {_DIST[quantity][2]} distribution over time",
                 subtitle=f"Dp (nm) over {_TIME_SUBTITLE}; colour {_DIST[quantity][1]}")


# --- time series ------------------------------------------------------------------------------

def time_series(results, keys, labels=None, log_y: bool = False, title: str | None = None) -> go.Figure:
    """One panel per series name (see results.series), one line per run; shared time axis.

    One run: each panel in its quantity's colour (config.QUANTITY_COLORS: SOA Green, sulfate
    Clay, ... as in composition(); gases and radicals as in config.SPECIES_STYLE). Two runs: A
    Charcoal solid, B Blue dashed, labelled "A" and "B" at the line ends; more runs: each its own
    colour and dash, labelled with its name.

    ``log_y`` sets the starting scale; the Linear/Log buttons switch every panel. On a log axis
    zeros (e.g. radicals at t = 0) are left out instead of drawn at the bottom of the axis.
    N2O5 is shown in ppt (config.DISPLAY_SCALE).
    """
    runs, labels = _runs(results, labels)
    keys = [keys] if isinstance(keys, str) else list(keys)
    multi = len(runs) > 1
    looks, names, tags = _run_looks(labels)
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    height = max(config.PLOT_HEIGHT, config.PLOT_HEIGHT_PER_PANEL * len(keys))
    panel_px = _panel_px(height, len(keys))
    quantities, scales, ys = [], {}, {}
    for row, key in enumerate(keys, start=1):
        name, unit = config.SERIES_LABELS.get(key, (key, ""))
        quantities.append((name, unit))
        panel, lines = [], []
        for j, (r, label) in enumerate(zip(runs, labels)):
            y = series(r, key) * config.DISPLAY_SCALE.get(key, 1.0)
            panel.append(y)
            look = looks[j] if multi else dict(color=_quantity_color(key), dash="solid",
                                               width=config.PLOT_LINE_WIDTH)
            ys[len(fig.data)] = (y, _positive(y))
            lines.append((r.time_h, y, look["color"], label, tags[j]))
            fig.add_trace(go.Scatter(
                x=r.time_h, y=_positive(y) if log_y else y, name=names[j] if multi else name,
                mode="lines", line=look, legendgroup=label, showlegend=multi and row == 1,
                hovertemplate=f"{label}<br>t = %{{x:.2f}} h<br>{name} = %{{y:.3g}} {unit}<extra></extra>"),
                row=row, col=1)
        axis = f"yaxis{row if row > 1 else ''}"
        scales[axis] = _y_scales(*panel)
        _apply(fig, axis, scales[axis][int(log_y)])
        if len(keys) > 1:
            fig.update_yaxes(title=f"{name}<br>({unit})" if unit else name, row=row, col=1)
        if multi and all(tags):
            _add_end_labels(fig, lines, log_y, panel_px, ys, row=row, bold=len(runs) == 2,
                            log_range=scales[axis][1].get("range"))
            _label_margin(fig, tags)
    fig.update_xaxes(**_time_axis(_time_range(*(r.time_h for r in runs))))
    nights = _shade_nights(fig, runs[0], panel_px)
    _add_scale_buttons(fig, scales, log_y, ys=ys)
    if title is None:
        title = _name(runs[0]) if not multi else _join([q[0] for q in quantities])
    if len(keys) == 1:
        name, unit = quantities[0]
        what = f"{name} ({unit})" if unit else name
        subtitle = f"{what} vs {_TIME_SUBTITLE}"
    else:
        joined = _join([q[0] for q in quantities])
        subtitle = f"{joined[:1].upper()}{joined[1:]} vs {_TIME_SUBTITLE}"
    return style(fig, title, height, subtitle=_with_night_note(subtitle, nights, multi))


def totals(results, labels=None, keys=config.DEFAULT_TOTALS, log_y: bool = False) -> go.Figure:
    """Particle number, SOA, sulfate, nitrate and dry mass over time (one or several runs)."""
    runs, labels = _runs(results, labels)
    title = f"{_name(runs[0])}: totals" if len(runs) == 1 else "Totals"
    return time_series(runs, keys, labels, log_y=log_y, title=title)


def composition(result, fraction: bool = False) -> go.Figure:
    """Dry particle mass by component over time, stacked (``fraction=True``: percent of dry mass).

    Components that stay zero are left out. The legend and the labels at the right edge give
    each component's value at the end of the run.
    """
    r = as_result(result)
    fig = make_subplots(rows=1, cols=1)
    t = np.asarray(r.time_h, dtype=float)
    layers = [(key, label, color, np.asarray(series(r, key), dtype=float))
              for key, label, color in config.COMPOSITION_LAYERS]
    layers = [layer for layer in layers if np.nanmax(np.abs(layer[3]), initial=0.0) > 0] or layers[:1]
    stack = np.nansum([layer[3] for layer in layers], axis=0)
    unit = "%" if fraction else "µg/m³"
    ends, total_end = [], float(stack[-1]) if stack.size else 0.0
    for n, (key, label, color, y) in enumerate(layers):
        end = float(y[-1])
        if fraction:
            end = 100.0 * end / total_end if total_end > 0 else 0.0
        ends.append(end)
        value = f"{end:.0f}" if fraction else f"{end:.3g}"
        name = f"{label} {value}" + (f" {unit} at {_hours(t[-1])} h" if n == len(layers) - 1 else "")
        fig.add_trace(go.Scatter(
            x=t, y=y, name=name, legendgroup=label, stackgroup="mass", mode="lines",
            line=dict(width=1, color="white", dash="solid"),
            fillcolor=_rgba(color, config.COMPOSITION_FILL_OPACITY),
            groupnorm="percent" if fraction else None,
            hovertemplate=f"{label}: %{{y:.3g}} {unit}<extra></extra>"))
    top = 100.0 if fraction else float(np.nanmax(stack)) if stack.size else 1.0
    fig.update_xaxes(**_time_axis(_time_range(t)))
    if fraction:
        fig.update_yaxes(range=[0, 100], **_linear_ticks(0, 100, unit=" %"))
    else:
        fig.update_yaxes(range=[0, 1.05 * top], **_linear_ticks(0, 1.05 * top))
    _band_labels(fig, layers, ends, top if fraction else 1.05 * top, t[-1], unit)
    # "night" above the plot, as on the app's composition card; with fraction=True the stack
    # fills the plot, so the band goes over it
    nights = _shade_nights(fig, r, label_above=True, over=fraction)
    fig.update_layout(hovermode="x unified")
    what = "Share of dry mass (%), stacked" if fraction else "Dry mass (µg/m³), stacked"
    return style(fig, f"{_name(r)}: particle composition",
                 subtitle=_with_night_note(f"{what}, vs {_TIME_SUBTITLE}", nights))


def gases(result, species=None, log_y: bool = True) -> go.Figure:
    """Gas mixing ratios (ppb). Default: every reported gas, the level's main ones visible."""
    r = as_result(result)
    default = config.DEFAULT_GASES.get(r.scenario.get("level"), ())
    return _species_plot(r, "gases_ppb", species, default, log_y, "Mixing ratio", "ppb", "gases")


def radicals(result, species=config.DEFAULT_RADICALS, log_y: bool = True) -> go.Figure:
    """OH, HO2 and NO3 concentrations (molecules per cm3)."""
    r = as_result(result)
    return _species_plot(r, "radicals_cm3", species, species, log_y, "Concentration", "molec/cm³",
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
    height = config.PLOT_HEIGHT + (120 if show_sza else 0)
    styles = _species_styles(j, held=())
    values, lines, ys = [], [], {}
    for name in j:
        label = config.SERIES_LABELS.get(name, (name, ""))[0]
        raw = series(r, name)
        values.append(_positive(raw) if log_y else raw)
        color, dash = styles[name]
        lines.append((r.time_h, raw, color, label, label))
        fig.add_trace(go.Scatter(x=r.time_h, y=values[-1], name=label, legendgroup=label, mode="lines",
                                 line=dict(color=color, dash=dash),
                                 hovertemplate=f"{label} = %{{y:.3g}} s⁻¹<extra></extra>"), row=1, col=1)
    y_axis = _log_axis(*values) if log_y else _linear_ticks(*_finite_range(values))
    _apply(fig, "yaxis", y_axis)
    _add_end_labels(fig, lines, log_y, _panel_px(height, 1) * (0.65 if show_sza else 1), ys, row=1,
                    log_range=y_axis.get("range"))
    fig.update_yaxes(title="J (s⁻¹)", row=1, col=1)
    if show_sza:
        fig.add_trace(go.Scatter(x=r.time_h, y=series(r, "sza_deg"), name="solar zenith angle",
                                 mode="lines", line=dict(color=config.CHARCOAL, dash="solid"),
                                 hovertemplate="SZA = %{y:.1f}°<extra></extra>"), row=2, col=1)
        fig.add_hline(y=config.NIGHT_SZA_DEG, line=_NIGHT_SZA_LINE, row=2, col=1)
        fig.update_yaxes(title="SZA", range=[0, 180], row=2, col=1, **_sza_ticks())
    fig.update_xaxes(**_time_axis(_time_range(r.time_h)))
    nights = _shade_nights(fig, r, _panel_px(height, 1) * (0.65 if show_sza else 1))
    _label_margin(fig, [line[4] for line in lines])
    mode = r.scenario.get("photolysis", {}).get("mode", "")
    return style(fig, f"{_name(r)}: sunlight (photolysis mode “{mode}”)", height,
                 subtitle=_with_night_note(f"J (s⁻¹) and solar zenith angle vs {_TIME_SUBTITLE}", nights))


def sunlight(days, j: str = "J_NO2") -> go.Figure:
    """J and solar zenith angle against local clock time for results of sunlight_over_day()."""
    days = [days] if isinstance(days, dict) else list(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    label = config.SERIES_LABELS.get(j, (j, ""))[0]
    height = config.PLOT_HEIGHT + 140
    lines, ys = [], {}
    short = _distinct_parts([day["label"] for day in days])
    for n, day in enumerate(days):
        x = day["local_start_h"] + np.asarray(day["time_h"])
        look = _style(n)
        lines.append((x, np.asarray(day[j], dtype=float), look["line"]["color"], day["label"], short[n]))
        fig.add_trace(go.Scatter(x=x, y=day[j], name=day["label"], legendgroup=day["label"], **look,
                                 hovertemplate=f"{label} = %{{y:.3g}} s⁻¹<extra>{day['label']}</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=day["sza_deg"], name=day["label"], legendgroup=day["label"],
                                 showlegend=False, **look,
                                 hovertemplate=f"SZA = %{{y:.1f}}°<extra>{day['label']}</extra>"),
                      row=2, col=1)
    fig.add_hline(y=config.NIGHT_SZA_DEG, line=_NIGHT_SZA_LINE, row=2, col=1)
    first = min(day["local_start_h"] for day in days)
    last = max(day["local_start_h"] + float(np.max(day["time_h"])) for day in days)
    step = 6 if last - first > 12 else 3
    ticks = list(range(int(np.ceil(first / step) * step), int(last) + 1, step))
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{h % 24:02d}:00" for h in ticks], range=[first, last])
    fig.update_yaxes(title=f"{label} (s⁻¹)", row=1, col=1,
                     **_linear_ticks(*_finite_range([line[1] for line in lines])))
    fig.update_yaxes(title="SZA", range=[0, 180], row=2, col=1, **_sza_ticks())
    if len(days) > 1:
        _add_end_labels(fig, lines, False, _panel_px(height, 1) * 0.6, ys, row=1, gap_px=16)
        _label_margin(fig, [line[4] for line in lines])
    return style(fig, "Sunlight over the day (TUV-x, clear sky)", height,
                 subtitle=f"{label} (s⁻¹) and solar zenith angle vs local clock time; "
                          "solar zenith angle above the dashed line: night")


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
    # Value labels next to the bars when only one or two classes carry mass (Level 1's surrogate)
    filled = np.flatnonzero(np.nanmax(part + gas, axis=0) > 0.01 * y_max)
    value_labels = len(filled) <= 2
    hover = "c* = 10<sup>%{x:.1f}</sup> µg/m³<br>%{y:.3g} µg/m³<extra>%{fullData.name}</extra>"

    def traces(i):
        coa_x = [log_coa[i]] * 2 if np.isfinite(log_coa[i]) else [None, None]
        return [go.Bar(x=x, y=part[i], name="particle phase", marker_color=config.PARTICLE_COLOR,
                       width=0.4, hovertemplate=hover),
                go.Bar(x=x, y=gas[i], name="gas phase", marker_color=config.GAS_COLOR, width=0.4,
                       hovertemplate=hover),
                go.Scatter(x=coa_x, y=[0, y_max], mode="lines", name="C_OA",
                           line=dict(color=config.REFERENCE_LINE_COLOR, dash=config.REFERENCE_DASH, width=1.2),
                           hoverinfo="skip")]

    def notes(i):
        out = []
        if np.isfinite(log_coa[i]):
            out.append(dict(x=log_coa[i], y=y_max, xanchor="left", yanchor="top", xshift=4, showarrow=False,
                            text=f"C_OA = {_fixed_decimals(coa[i])} µg/m³", font=_label_font(config.CHARCOAL)))
        if value_labels:
            for k in filled:
                for value, base, label, bold in ((part[i, k], 0.0, "particle", True),
                                                 (gas[i, k], part[i, k], "gas", False)):
                    if value > 0.02 * y_max:
                        out.append(dict(x=x[k] - 0.2, y=base + value / 2, xanchor="right", yanchor="middle",
                                        xshift=-4, showarrow=False, text=f"{label} {value:.3g}",
                                        font=_label_font(config.CHARCOAL if bold else config.TEXT_2, bold)))
        return out

    last = t.size - 1
    fig = go.Figure(traces(last))
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    text = [f"10<sup>{int(k)}</sup>" for k in ticks]
    text[-1] += " µg/m³"
    fig.update_xaxes(range=[lo, hi], tickvals=ticks, ticktext=text)
    fig.update_yaxes(range=[0, y_max], **_linear_ticks(0, y_max))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    _add_time_slider(fig, t, [(traces(i), [0, 1, 2]) for i in range(t.size)], notes=notes)
    return style(fig, f"{_name(r)}: organic mass by volatility class",
                 subtitle="Organic mass (µg/m³) vs c* at 298 K (µg/m³), split into particle and gas")


def partitioning(coa_ugm3=config.PARTITIONING_COA_UGM3, log10_cstar=config.PARTITIONING_LOG10_CSTAR,
                 start_coa_ugm3=config.PARTITIONING_START_COA_UGM3,
                 highlight_cstar: float | None = None, highlight_label: str | None = None) -> go.Figure:
    """How the organic aerosol mass C_OA sets each volatility class's particle share (no model run).

    Equilibrium absorptive partitioning (Pankow, 1994), F_p = 1 / (1 + c*/C_OA): one bar per
    c* class, particle share (green) under gas share (blue). The slider (and play button) sets
    C_OA; the dashed line marks c* = C_OA, where a class is half in the particles.
    ``highlight_cstar`` outlines one class and ``highlight_label`` names it under its tick.
    """
    x = np.asarray(log10_cstar, dtype=float)
    coa = np.asarray(coa_ugm3, dtype=float)
    marked = np.zeros(x.size, bool) if highlight_cstar is None else np.isclose(x, np.log10(highlight_cstar))
    outline = dict(color=config.CHARCOAL, width=np.where(marked, 3, 0).tolist())

    def traces(c):
        share = 100.0 / (1.0 + 10.0 ** x / c)
        hover = "c* = 10<sup>%{x:.0f}</sup> µg/m³: %{y:.0f} %<extra>%{fullData.name}</extra>"
        return [go.Bar(x=x, y=share, name="in the particles", width=0.6, hovertemplate=hover,
                       marker=dict(color=config.PARTICLE_COLOR, line=outline)),
                go.Bar(x=x, y=100.0 - share, name="in the gas", width=0.6, hovertemplate=hover,
                       marker=dict(color=config.GAS_COLOR, line=outline)),
                go.Scatter(x=[np.log10(c)] * 2, y=[0, 100], mode="lines",
                           name="c* = C_OA: half in the particles",
                           line=dict(color=config.REFERENCE_LINE_COLOR, dash=config.REFERENCE_DASH, width=1.2),
                           hoverinfo="skip")]

    start = int(np.argmin(np.abs(np.log10(coa / start_coa_ugm3))))
    fig = go.Figure(traces(coa[start]))
    lo, hi = min(x.min(), np.log10(coa.min())) - 0.6, max(x.max(), np.log10(coa.max())) + 0.6
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
    labels = [f"10<sup>{int(k)}</sup>" for k in ticks]
    labels[-1] += " µg/m³"
    if highlight_label and marked.any():
        labels = [f"{t}<br><b>{highlight_label}</b>" if np.isclose(k, x[marked][0]) else t
                  for t, k in zip(labels, ticks)]
    fig.update_xaxes(range=[lo, hi], tickvals=ticks, ticktext=labels)
    fig.update_yaxes(range=[0, 100], **_linear_ticks(0, 100, step=25, unit=" %"))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    _add_time_slider(fig, coa, [(traces(c), [0, 1, 2]) for c in coa], active=start,
                     readout=lambda c: f"C_OA = {c:g} µg/m³", ticks=_decade_steps(coa, "µg/m³"),
                     pad_t=62 if highlight_label else 44)              # room for the highlight label
    return style(fig, "Equilibrium partitioning: F<sub>p</sub> = 1 / (1 + c*/C<sub>OA</sub>)",
                 subtitle="Share of each class (%) vs saturation concentration c* (µg/m³)")


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

    def row_label(c):           # species code in JetBrains Mono, under it formula and log10 c*
        return (f'<span style="font-family:{config.PLOT_MONO_FONT_FAMILY};font-size:12px;color:{config.CHARCOAL}">'
                f'{c["species"]}</span><br><span style="font-size:10.5px;color:{config.TEXT_2}">'
                f'{_formula(c["formula"])} · log₁₀c* {c["log10_cstar"]:.2f}</span>')

    fig = go.Figure()
    # "no N" bars also hatched: Clay and Green are a red-green pair, so colour alone is not enough
    for flag, label, color in ((True, "contains N (nitrate, PAN)", config.SERIES_COLORS[1]),
                               (False, "no N", config.PARTICLE_COLOR)):
        sel = [c for c, f in zip(rows, has_n) if f == flag]
        fig.add_trace(go.Bar(
            y=[row_label(c) for c in sel], x=[c["particle_ugm3"] for c in sel],
            orientation="h", name=label,
            marker=dict(color=color, pattern=dict(shape="") if flag else config.NO_N_PATTERN),
            text=[f"{c['particle_ugm3']:.2f}" for c in sel], textposition="outside", cliponaxis=False,
            textfont=dict(family=config.PLOT_MONO_FONT_FAMILY, size=config.PLOT_LABEL_FONT_SIZE,
                          color=config.CHARCOAL),
            customdata=[[c["species"], c["formula"], c["log10_cstar"]] for c in sel],
            hovertemplate=("%{customdata[0]} · %{customdata[1]}<br>particle: %{x:.3g} µg/m³"
                           "<br>log₁₀ c* = %{customdata[2]:.2f}<extra></extra>")))
    fig.update_yaxes(categoryorder="array", categoryarray=[row_label(c) for c in rows],
                     showline=False, ticks="", ticklabelstandoff=10)
    fig.update_xaxes(visible=False, range=[0, 1.12 * max(c["particle_ugm3"] for c in rows)])
    fig.update_layout(bargap=0.55, barmode="overlay")
    height = max(config.PLOT_HEIGHT, 38 * len(rows) + 140)
    return style(fig, f"{_name(r)}: top {len(rows)} SOA species", height,
                 subtitle=f"Particle-phase mass at the end of the run (µg/m³); "
                          f"N-containing: {100 * share_n:.0f}% of their mass")


# --- helpers: series and styles ---------------------------------------------------------------

def _species_plot(r, family, species, visible, log_y, y_title, unit, what) -> go.Figure:
    values = getattr(r, family)
    if not values:
        return _message_figure(f"No {what} in this run (Level 1 has no gas chemistry).")
    show_all = species is None
    names = list(values) if show_all else ([species] if isinstance(species, str) else list(species))
    held = set(r.scenario.get("hold_constant") or ())
    styles = _species_styles(names, held)
    fig = make_subplots(rows=1, cols=1)
    shown, lines, ys = [], [], {}
    for name in names:
        label = config.SERIES_LABELS.get(name, (name, ""))[0] + (" (held)" if name in held else "")
        raw = series(r, name)
        is_visible = not show_all or name in visible
        if is_visible:
            shown.append(_positive(raw) if log_y else raw)
        color, dash = styles[name]
        ys[len(fig.data)] = (raw, _positive(raw))
        lines.append((r.time_h, raw, color, name, label, is_visible))
        fig.add_trace(go.Scatter(
            x=r.time_h, y=_positive(raw) if log_y else raw, name=label, legendgroup=name, mode="lines",
            line=dict(color=color, dash=dash), visible=True if is_visible else "legendonly",
            hovertemplate=f"{label} = %{{y:.3g}} {unit}<extra></extra>"))
    scales = {"yaxis": _y_scales(*(shown or [ys[k][0] for k in ys]))}
    # The log range is fixed (ticks at both ends), so a legend click does not rescale it: it also
    # reaches up to the largest value and down to the peak of each hidden line, so a line shown
    # from the legend is in view (at least its top part).
    hidden = [np.asarray(line[1], dtype=float) for line in lines if not line[5]]
    hidden = [float(y[np.isfinite(y) & (y > 0)].max()) for y in hidden if (np.isfinite(y) & (y > 0)).any()]
    if hidden and shown:
        scales["yaxis"] = (scales["yaxis"][0], {"autorange": True, **_log_axis(*shown, hidden)})
    _apply(fig, "yaxis", scales["yaxis"][int(log_y)])
    _add_end_labels(fig, [(t, y, c, group, label) for t, y, c, group, label, _ in lines], log_y,
                    _panel_px(config.PLOT_HEIGHT, 1), ys, row=1,
                    visible=[line[5] for line in lines], log_range=scales["yaxis"][1].get("range"))
    fig.update_xaxes(**_time_axis(_time_range(r.time_h)))
    _add_scale_buttons(fig, scales, log_y, ys=ys)
    nights = _shade_nights(fig, r)
    _label_margin(fig, [line[4] for line in lines])
    subtitle = f"{y_title} ({unit}) vs {_TIME_SUBTITLE}"
    if show_all:
        subtitle += "; click a legend entry to show or hide it"
    return style(fig, f"{_name(r)}: {what}", subtitle=_with_night_note(subtitle, nights))


def _species_styles(names, held):
    """name -> (colour, dash): config.SPECIES_STYLE (held NO2: config.HELD_STYLE), then the
    fallback styles not used yet in this plot."""
    out = {}
    for name in names:
        if name in held:
            out[name] = config.HELD_STYLE
        elif name in config.SPECIES_STYLE:
            out[name] = config.SPECIES_STYLE[name]
    spare = [s for s in config.SPECIES_FALLBACK_STYLES if s not in out.values()]
    for name in names:
        if name not in out:
            out[name] = spare.pop(0) if spare else (_color(len(out)), config.SERIES_DASHES[len(out) % 9])
    return out


def _quantity_color(key):
    if key in config.QUANTITY_COLORS:
        return config.QUANTITY_COLORS[key]
    if key in config.SPECIES_STYLE:
        return config.SPECIES_STYLE[key][0]
    return _color(0)


def _run_looks(labels):
    """Line, legend name and line-end label of each run. Two runs: A and B as in the web app's
    compare view (legend "A  name"); more: the series colours and dashes, labelled by name."""
    if len(labels) == 2:
        looks = [dict(color=config.RUN_A_COLOR, dash="solid", width=config.PLOT_COMPARE_LINE_WIDTH),
                 dict(color=config.RUN_B_COLOR, dash=config.RUN_B_DASH, width=config.PLOT_COMPARE_LINE_WIDTH)]
        names = [label if label == tag else
                 f'<span style="color:{look["color"]}"><b>{tag}</b></span>  {label}'
                 for tag, label, look in zip("AB", labels, looks)]
        return looks, names, ["A", "B"]
    looks = [dict(color=_color(j), dash=config.SERIES_DASHES[j % len(config.SERIES_DASHES)],
                  width=config.PLOT_LINE_WIDTH) for j in range(len(labels))]
    short = all(len(_plain(label)) <= _LABEL_MAX_CHARS for label in labels)
    return looks, list(labels), [label if short else None for label in labels]


def _dist(r, quantity):
    if quantity not in _DIST:
        raise ValueError(f"quantity must be one of {list(_DIST)}")
    z = np.asarray(getattr(r, _DIST[quantity][0]), dtype=float)
    return np.asarray(r.time_h, dtype=float), np.asarray(r.diameter_nm, dtype=float), z


def _dist_trace(d, y, name, look):
    return go.Scatter(x=d, y=y, name=name, **look,
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


def _every(t, step_h):
    """Indices of the first output at or after each multiple of step_h (excluding t = 0)."""
    targets = np.arange(step_h, t[-1] + 1e-9, step_h)
    return sorted({int(i) for i in np.searchsorted(t, targets - 1e-9) if i < t.size})


def _hours(t):
    return f"{round(float(t), 2) + 0.0:.2f}".rstrip("0").rstrip(".")


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


def _style(i, markers: bool = False) -> dict:
    """Line of series ``i``: its own colour and dash (Scatter keywords); ``markers=True`` adds
    its own marker symbol, sparse (for figures of your own; the plots draw no markers).

    Colour-blind safe: the series stay apart on dash alone.
    """
    look = dict(mode="lines", line=dict(color=_color(i),
                                        dash=config.SERIES_DASHES[i % len(config.SERIES_DASHES)]))
    if markers:
        symbols = config.SERIES_MARKERS                  # one longer than colours and dashes
        look.update(mode="lines+markers", marker=dict(symbol=symbols[i % len(symbols)],
                                                      size=config.PLOT_MARKER_SIZE, maxdisplayed=10))
    return look


def _positive(y):
    """Values <= 0 become nan, which a log axis leaves out as a gap."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha:g})"


def _ink(color):
    """Text colour for a series colour (Slate, Gold and Green are not for text)."""
    return config.INK.get(color, color)


def _label_font(color, bold=True):
    return dict(color=_ink(color), size=config.PLOT_LABEL_FONT_SIZE, weight=600 if bold else 400)


def _formula(formula):
    """C9H13NO6 -> C₉H₁₃NO₆."""
    return formula.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _plain(text):
    return re.sub(r"<[^>]+>", "", str(text))


def _join(names):
    """"A, b and c": names after the first in lower case (sentence case), except acronyms and
    formulas ("SOA", "N₂O₅", "J(NO₂)": their second character is not a lower-case letter)."""
    names = [names[0]] + [n[:1].lower() + n[1:] if n[1:2].islower() else n for n in names[1:]]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _distinct_parts(labels):
    """Line-end labels for "site, date" legend names: the site if all dates are the same, the
    date if all sites are, else the whole name."""
    parts = [label.rsplit(", ", 1) if ", " in label else [label, ""] for label in labels]
    if len({p[1] for p in parts}) == 1 and len({p[0] for p in parts}) == len(parts):
        return [p[0] for p in parts]
    if len({p[0] for p in parts}) == 1 and len({p[1] for p in parts}) == len(parts):
        return [p[1] for p in parts]
    return list(labels)


def _with_night_note(subtitle, nights, first_run_only=False):
    if not nights:
        return subtitle
    return subtitle + ("; shaded bands: night in the first run (SZA > 90°)" if first_run_only
                       else "; shaded bands: night (SZA > 90°)")


# --- helpers: axes and ticks ------------------------------------------------------------------

def _apply(fig, axis, settings):
    """Set ``settings`` (attribute -> value) on layout axis ``axis`` ("yaxis", "yaxis2", ...)."""
    for key, value in settings.items():
        fig.layout[axis][key] = value


def _diameter_axis(log_range):
    """Log diameter axis on whole decades, ticks on decades only, the first one "… nm"."""
    lo, hi = int(np.floor(log_range[0] + 1e-9)), int(np.ceil(log_range[1] - 1e-9))
    hi = max(hi, lo + 1)
    vals = [10.0 ** e for e in range(lo, hi + 1)]
    text = [f"{v:g}" if v < 1e4 else f"10<sup>{e}</sup>" for v, e in zip(vals, range(lo, hi + 1))]
    text[0] += " nm"
    return dict(type="log", range=[lo, hi], tickmode="array", tickvals=vals, ticktext=text)


def _time_axis(time_range):
    """Time axis in hours: 4 to 7 round ticks, "0 h" on the first and the unit on the last."""
    t0, t1 = time_range
    span = max(t1 - t0, 1e-9)
    step = next((s for s in (0.1, 0.2, 0.25, 0.5, 1, 2, 3, 6, 12, 24, 48) if span / s <= 6 + 1e-9), 96)
    ticks = np.arange(np.ceil(t0 / step - 1e-9), np.floor(t1 / step + 1e-9) + 1) * step
    text = [_hours(float(v)) for v in ticks]
    if text:
        text[0] += " h"
        if len(text) > 1 and abs(ticks[-1] - t1) < 1e-6 * span:
            text[-1] += " h"
    return dict(range=[t0, t1], tickmode="array", tickvals=[float(v) for v in ticks], ticktext=text)


def _time_range(*times):
    """[first, last] time: the axis ends at the data."""
    return [min(float(np.min(t)) for t in times), max(float(np.max(t)) for t in times)]


def _sza_ticks():
    return dict(tickmode="array", tickvals=[0, 45, 90, 135, 180], ticktext=["0", "45", "90", "135", "180°"])


def _finite_range(arrays):
    v = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays]) if arrays else np.array([])
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)


def _linear_ticks(lo, hi, step=None, unit=""):
    """Three to five round ticks between lo and hi, labelled as in the app ("2,000", "4×10⁴");
    ``unit`` goes on the top tick."""
    if not np.isfinite(lo) or not np.isfinite(hi) or not hi > lo:
        return dict(tickmode="auto", tickvals=None, ticktext=None)
    step = step or _nice_step((hi - lo) / 3.5)
    ticks = np.arange(np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9) + 1) * step
    if ticks.size < 3 and not unit:              # at least three ticks
        step = _nice_step(step / 2.5)
        ticks = np.arange(np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9) + 1) * step
    ticks = [float(f"{v:.12g}") for v in ticks]
    text = [_tick_label(v) for v in ticks]
    if unit and text:
        text[-1] += unit
    return dict(tickmode="array", tickvals=ticks, ticktext=text)


def _fixed_y(z, log_y):
    """Fixed y range and ticks for size distributions (which must not rescale while animating)."""
    top = float(np.nanmax(z)) if np.size(z) else 0.0
    if not top > 0:
        return {}
    if log_y:
        return {**_log_axis([top * 1e-4, top]), "range": [np.log10(top) - 4, np.log10(top) + 0.1]}
    return dict(range=[0, 1.05 * top], **_linear_ticks(0, 1.05 * top))


def _y_scales(*arrays, fixed=None):
    """(linear, log) settings for one y axis: fixed ranges from ``fixed`` (size distributions)
    or autorange, both with the app's tick labels."""
    if fixed is not None:
        return ({**_fixed_y(fixed, False), "type": "linear", "autorange": False},
                {**_fixed_y(fixed, True), "autorange": False})
    lo, hi = _finite_range(arrays)
    if hi - lo <= 1e-9 * max(abs(hi), 1e-300):    # constant: from 0 to twice the value
        top = 2 * hi if hi > 0 else 1.0
        lin = {**_linear_ticks(0, top), "range": [0, top], "autorange": False}
    else:
        lin = {**_linear_ticks(lo, hi), "autorange": True}
    return ({**lin, "type": "linear"}, {"autorange": True, **_log_axis(*arrays)})


def _log_axis(*arrays) -> dict:
    """Log axis with readable tick labels for the range of the positive values in ``arrays``.

    Narrow ranges get evenly spaced ticks, about one decade 1, 2 and 5 per decade, up to
    config.LOG_AXIS_DETAIL_DECADES 1 and 3, wider ranges powers of ten only, every 2nd or 3rd
    decade so that 3 to config.LOG_AXIS_MAX_TICKS remain. The range (log10 units, autorange
    off) is snapped outward to the nearest tick, so the top and bottom data have a tick beyond
    them, as in the app (log axes on whole decades).
    """
    v = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays]) if arrays else np.array([])
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return dict(type="log", tickmode="auto", tickvals=None, ticktext=None)
    lo, hi = float(v.min()), float(v.max())
    span = np.log10(hi / lo)
    lo_e, hi_e = int(np.floor(np.log10(lo) + 1e-9)), int(np.ceil(np.log10(hi) - 1e-9))
    hi_e = max(hi_e, lo_e + 1)
    decades = np.arange(lo_e, hi_e + 1)
    inner = [int(e) for e in decades if lo <= 10.0 ** e * (1 + 1e-9) and 10.0 ** e <= hi * (1 + 1e-9)]
    decade_ticks = False
    if span > config.LOG_AXIS_DETAIL_DECADES and len(inner) < 3:
        span = config.LOG_AXIS_DETAIL_DECADES          # too few decades in view: 1 and 3 as well
    if span < config.LOG_AXIS_LINEAR_BELOW_DECADES:
        step = _nice_step((hi - lo) / 3) if hi > lo else 10.0 ** np.floor(np.log10(lo))
        ticks = [t for t in np.arange(np.floor(lo / step + 1e-9), np.ceil(hi / step - 1e-9) + 1) * step if t > 0]
    elif span <= 1.2:
        ticks = [m * 10.0 ** e for e in decades for m in (1, 2, 5)]
    elif span <= config.LOG_AXIS_DETAIL_DECADES:
        ticks = [m * 10.0 ** e for e in decades for m in (1, 3)]
    else:
        # every k-th decade from a tick at or below the data to one at or above it, k as small as
        # keeps at most LOG_AXIS_MAX_TICKS; of the k choices of which decades, the one that
        # extends the axis least beyond the data (even exponents on a tie)
        def decade_ticks_for(k, o):
            first = lo_e - (lo_e - o) % k
            return list(range(first, hi_e + k, k))          # the last one at or above hi_e

        every = next(k for k in range(1, len(decades) + 1)
                     if min(len(decade_ticks_for(k, o)) for o in range(k)) <= config.LOG_AXIS_MAX_TICKS)
        offset = min(range(every), key=lambda o: (decade_ticks_for(every, o)[-1] - decade_ticks_for(every, o)[0],
                                                  o % 2 != 0))
        ticks = [10.0 ** e for e in decade_ticks_for(every, offset)]
        decade_ticks = True
    ticks = [float(f"{t:.12g}") for t in ticks]
    below = [t for t in ticks if t <= lo * (1 + 1e-9)]
    above = [t for t in ticks if t >= hi * (1 - 1e-9)]
    bottom, top = (below[-1] if below else ticks[0]), (above[0] if above else ticks[-1])
    ticks = [t for t in ticks if bottom <= t <= top]
    log_range = [float(np.log10(bottom)), float(np.log10(top))]
    if not log_range[1] > log_range[0]:                # a constant: half a decade either side
        log_range = [float(np.log10(lo)) - 0.5, float(np.log10(hi)) + 0.5]
    small = 1.0 if any(t < 1e-3 for t in ticks) else 1e-3   # all of one axis' small ticks alike
    return dict(type="log", tickmode="array", tickvals=ticks, autorange=False,
                range=log_range, ticktext=[_tick_label(t, log=decade_ticks, power_below=small) for t in ticks])


def _nice_step(x):
    exponent = np.floor(np.log10(x))
    fraction = x / 10.0 ** exponent
    return (1 if fraction < 1.5 else 2 if fraction < 3.5 else 5 if fraction < 7.5 else 10) * 10.0 ** exponent


def _tick_label(value, log=False, power_below=1e-3):
    """Tick labels as in the app: '2,000' below 10⁴, '6×10⁴' from 10⁴ up, '0.004' for small
    numbers, '10⁻⁴' below ``power_below`` (0.001). ``log=True`` (log axes ticked on decades):
    powers of ten below 1 as well ('10⁻²')."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value)) + 1e-9))
    mantissa = float(f"{value / 10.0 ** exponent:.3g}")
    if abs(mantissa) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    power = abs(value) >= 1e4 or abs(value) < power_below or (log and abs(value) < 1 and abs(mantissa) == 1)
    if not power:
        return f"{value:,.6g}"
    return f"10<sup>{exponent}</sup>" if mantissa == 1 else f"{mantissa:g}×10<sup>{exponent}</sup>"


def _fixed_decimals(x):
    """'5.13', '9.00', '16.6', '230': two decimals below 10, one below 100, none above."""
    return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}" if abs(x) < 100 else f"{x:.0f}"


def _round_down_1(x):
    """x rounded down to one significant figure (0.39 -> 0.3, 0.86 -> 0.8)."""
    if not x > 0:
        return x
    unit = 10.0 ** np.floor(np.log10(x) + 1e-9)
    return float(f"{np.floor(x / unit + 1e-9) * unit:.12g}")


def _round2(x):
    return float(f"{x:.2g}")


def _decade_steps(values, unit):
    """Step index -> label for the values that are whole decades (labels under a slider); the
    unit on the last."""
    logs = np.log10(np.asarray(values, dtype=float))
    idx = [i for i, v in enumerate(logs) if abs(v - round(v)) < 1e-9]
    return {i: f"{values[i]:g}" + (f" {unit}" if i == idx[-1] else "") for i in idx}


# --- helpers: labels, bands, controls -----------------------------------------------------------

def _label_trace(x, y, text, color, group, position="middle right", visible=True, bold=True, xaxis=None,
                 yaxis=None):
    """A label drawn as a text-only trace in the line's legend group, so a legend click hides it
    with its line (and the Linear/Log buttons can move it)."""
    trace = go.Scatter(x=[x], y=[y], mode="text", text=[text], textposition=position, name=_plain(text).strip(),
                       textfont=_label_font(color, bold), legendgroup=group, showlegend=False,
                       hoverinfo="skip", cliponaxis=False, visible=True if visible else "legendonly")
    if xaxis:
        trace.update(xaxis=xaxis, yaxis=yaxis)
    return trace


def _add_end_labels(fig, lines, log_y, panel_px, ys, row=1, visible=None, bold=True, gap_px=_LABEL_GAP_PX,
                    log_range=None):
    """A label at the end of each line (``lines``: (x, y, colour, legend group, text)), a small
    gap after the line's end, moved apart (to at least ``gap_px``) where they would overlap;
    ``ys`` gets each label's (linear, log) y for the buttons. ``log_range``: the log axis'
    range (log10 units) if it is fixed, so the spacing is reckoned in its pixels."""
    if not lines:
        return
    visible = visible or [True] * len(lines)
    xs, yv = [], []
    for x, y, *_ in lines:                 # the last point a log axis can show (else the last one)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y) & (y > 0)
        if not ok.any():
            ok = np.isfinite(y)
        k = int(np.flatnonzero(ok)[-1]) if ok.any() else -1
        xs.append(float(np.asarray(x)[k]) if k >= 0 else None)
        yv.append(float(y[k]) if k >= 0 else None)
    values = [np.asarray(line[1], dtype=float) for line, v in zip(lines, visible) if v]
    spread = {scale: _spread_1d(yv, values, scale, panel_px, visible, gap_px,
                                axis_range=log_range if scale else None) for scale in (False, True)}
    xaxis, yaxis = ("x", "y") if row == 1 else (f"x{row}", f"y{row}")
    for n, (x, y, color, group, text) in enumerate(lines):
        if xs[n] is None:
            continue
        ys[len(fig.data)] = ([spread[False][n]], [spread[True][n]])
        fig.add_trace(_label_trace(xs[n], spread[log_y][n], _END_LABEL_LEAD + text, color, group,
                                   visible=visible[n], bold=bold, xaxis=xaxis, yaxis=yaxis))


def _spread_1d(ys, panel_values, log, panel_px, visible, gap_px=_LABEL_GAP_PX, axis_range=None):
    """Label heights: the line ends, moved apart to at least ``gap_px`` (visible labels only),
    in the pixels of ``axis_range`` (axis units) if given, else of the values' range."""
    v = np.concatenate([np.ravel(a) for a in panel_values]) if panel_values else np.array([])
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if v.size == 0:
        return ys
    tr = (lambda a: np.log10(a)) if log else (lambda a: a)
    back = (lambda a: 10.0 ** a) if log else (lambda a: a)
    lo, hi = float(tr(v.min())), float(tr(v.max()))
    if not hi > lo:
        return ys
    gap = gap_px / panel_px * ((axis_range[1] - axis_range[0]) if axis_range is not None else hi - lo)
    idx = [i for i, y in enumerate(ys) if y is not None and visible[i] and (y > 0 or not log)]
    pos = {i: float(tr(ys[i])) for i in idx}
    order = sorted(idx, key=lambda i: pos[i])
    for a, b in zip(order, order[1:]):
        pos[b] = max(pos[b], pos[a] + gap)
    overflow = pos[order[-1]] - hi if order else 0.0
    if overflow > 0:                               # pushed over the top: shift down, keep the gaps
        for i in order:
            pos[i] -= overflow
        for a, b in zip(order[::-1], order[::-1][1:]):
            pos[b] = min(pos[b], pos[a] - gap)
    out = list(ys)
    for i in idx:
        out[i] = float(back(pos[i]))
    return out


def _spread_2d(x_log, y, scales, tags, lift_px=12):
    """Peak labels (x in log10 nm, y in data units) ``lift_px`` above each peak, for each of
    ``scales`` ((y range, log) per Linear/Log button). Two peaks closer than a label width
    (in x and, on either scale, in y) get their labels side by side at the higher one: the
    left one's ends at its peak, the right one starts there. Others that overlap move down
    (px estimates). Returns (y per scale, text position per label)."""
    px_x = _PLOT_WIDTH_PX / 4.0                      # about four decades across
    widths = [len(_plain(tag)) * _CHAR_PX + 8 for tag in tags]
    where = ["top center"] * len(y)

    def per_scale(y_range, log):
        lo, hi = y_range
        tr = (lambda a: np.log10(a) if a > 0 else lo) if log else (lambda a: a)
        px_y = _panel_px(config.PLOT_HEIGHT, 1) / max(hi - lo, 1e-12)
        return [min(tr(v) + lift_px / px_y, hi - 18 / px_y) for v in y], px_y, tr

    lifted = [per_scale(rng, log) for rng, log in scales]
    pairs = set()
    for i in range(len(y)):
        for k in range(i + 1, len(y)):
            near_x = abs(x_log[i] - x_log[k]) * px_x < (widths[i] + widths[k]) / 2
            if near_x and any(abs(yy[i] - yy[k]) * px_y < _LABEL_GAP_PX + 2 for yy, px_y, _ in lifted):
                pairs.add((i, k))
    taken = set()
    for i, k in sorted(pairs):
        if i in taken or k in taken:
            continue
        taken.update((i, k))
        left, right = (i, k) if (x_log[i], i) <= (x_log[k], k) else (k, i)
        where[left], where[right] = "top left", "top right"
    out = []
    for (y_range, log), (yy, px_y, _) in zip(scales, lifted):
        yy = list(yy)
        for i, k in sorted(pairs):
            if where[i] != "top center" and where[k] != "top center":
                yy[i] = yy[k] = max(yy[i], yy[k])
        placed, ys = [], [None] * len(y)
        for i in sorted(range(len(y)), key=lambda n: -yy[n]):
            yi = yy[i]
            if where[i] == "top center":
                for _ in range(len(y)):
                    clash = [p for p in placed if abs(p[0] - x_log[i]) * px_x < (widths[i] + p[2]) / 2
                             and abs(p[1] - yi) * px_y < _LABEL_GAP_PX + 2]
                    if not clash:
                        break
                    yi = min(p[1] for p in clash) - (_LABEL_GAP_PX + 2) / px_y
            placed.append((x_log[i], yi, widths[i]))
            ys[i] = float(10.0 ** yi) if log else float(yi)
        out.append(ys)
    return out, where


def _label_margin(fig, texts):
    """Right margin wide enough for the labels at the line ends."""
    texts = [t for t in texts if t]
    if texts:
        fig.update_layout(margin=dict(r=int(16 + 10 + 6 + _CHAR_PX * max(len(_plain(t)) for t in texts))))


def _band_labels(fig, layers, ends, y_top, t_end, unit):
    """A white label box at the right edge of each composition band: "■ SOA 4.94"."""
    bottoms = np.cumsum([0.0] + ends[:-1])
    mids = [b + e / 2 for b, e in zip(bottoms, ends)]
    keep = [i for i, e in enumerate(ends) if e > 0]
    pos = _spread_1d([mids[i] for i in keep], [np.array([0.0, y_top])], False,
                     _panel_px(config.PLOT_HEIGHT, 1), [True] * len(keep))
    for i, y in zip(keep, pos):
        key, label, color, _ = layers[i]
        value = f"{ends[i]:.0f} %" if unit == "%" else f"{ends[i]:.3g}"
        fig.add_annotation(x=t_end, y=y, xref="x", yref="y", xanchor="right", yanchor="middle", xshift=-4,
                           showarrow=False, bgcolor="white", bordercolor=config.LINE, borderwidth=1,
                           borderpad=2, font=dict(size=config.PLOT_NOTE_FONT_SIZE, color=config.CHARCOAL,
                                                  weight=600),
                           text=f'<span style="color:{color}">■</span> {label} {value}')


_NIGHT_ABOVE = "night above the plot"   # annotation name: style() makes room for it
_NIGHT_LABEL_PX = (34, 15)              # "night": width and height (px, estimate)


def _shade_nights(fig, r, panel_px=None, label_above=False, over=False):
    """Sand bands at night, the word "night" in each (first panel): as high as it fits with
    config.NIGHT_LABEL_CLEARANCE_PX to every line, else (and with ``label_above``) just above
    the plot. ``over``: the band goes over the traces, lighter (for plots the traces fill).
    ``panel_px``: the first panel's height (px, estimate). Returns whether there were any."""
    intervals = night_intervals(r)
    if not intervals:
        return False
    t = np.asarray(r.time_h, dtype=float)
    span = float(t[-1] - t[0]) or 1.0
    for t0, t1 in intervals:
        fig.add_vrect(x0=t0, x1=t1, fillcolor=config.NIGHT_OVERLAY_COLOR if over else config.NIGHT_COLOR,
                      line_width=0, layer="above" if over else "below", row="all", col=1)
        if (t1 - t0) / span > 0.06:
            y = None if label_above else _night_label_height(fig, t0, t1, span, panel_px or _panel_px(
                config.PLOT_HEIGHT, 1))
            font = dict(size=config.PLOT_NOTE_FONT_SIZE, color=config.TEXT_2)
            if y is None:
                fig.add_annotation(x=(t0 + t1) / 2, xref="x", y=1, yref="y domain", yanchor="bottom", yshift=1,
                                   showarrow=False, text=config.NIGHT_LABEL, font=font, name=_NIGHT_ABOVE)
            else:
                fig.add_annotation(x=(t0 + t1) / 2, xref="x", y=y, yref="y domain", yanchor="middle",
                                   showarrow=False, text=config.NIGHT_LABEL, font=font)
    return True


def _night_label_height(fig, t0, t1, span, panel_px):
    """Height (0 to 1 of the first panel) for the "night" label of the band [t0, t1]: the
    highest at which every visible line of the panel stays at least
    config.NIGHT_LABEL_CLEARANCE_PX from the label box (and its sides); None if there is none."""
    axis = fig.layout.yaxis
    log = axis.type == "log"
    lines = []
    for tr in fig.data:
        if not (tr.type == "scatter" and tr.mode == "lines" and tr.visible in (None, True)
                and tr.yaxis in (None, "y") and tr.y is not None):
            continue
        x, y = np.asarray(tr.x, dtype=float), np.asarray(tr.y, dtype=float)
        lines.append((x, np.log10(np.where(y > 0, y, np.nan)) if log else y))
    allv = np.concatenate([y for _, y in lines]) if lines else np.array([])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 0.97
    if axis.range is not None and axis.autorange is False:
        lo, hi = (float(v) for v in axis.range)
    else:                                            # autorange: the data, and Plotly's ~5 % pad
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
    if not hi > lo:
        return 0.97
    width_px, height_px = _NIGHT_LABEL_PX
    clear_px = config.NIGHT_LABEL_CLEARANCE_PX
    mid = (t0 + t1) / 2
    half = (width_px / 2 + clear_px) / _PLOT_WIDTH_PX * span
    grid = np.linspace(mid - half, mid + half, 15)
    inside = []
    for x, y in lines:
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        xs, ys = x[ok], y[ok]
        near = grid[(grid >= xs[0]) & (grid <= xs[-1])]
        pts = np.concatenate([np.interp(near, xs, ys), ys[np.abs(xs - mid) <= half]])
        inside.append((pts - lo) / (hi - lo) * panel_px)  # px above the bottom of the panel
    inside = np.concatenate(inside) if inside else np.array([])
    need = height_px / 2 + clear_px
    for centre in np.arange(panel_px - height_px / 2 - 2, height_px / 2 + 2, -2.0):
        if inside.size == 0 or np.min(np.abs(inside - centre)) >= need:
            return float(centre / panel_px)
    return None


def _caption(fig, text):
    """A line under the plot (inside the figure), as the captions in the app's cards."""
    fig.add_annotation(text=text, x=0, xref="paper", xanchor="left", y=0, yref="paper", yanchor="top",
                       yshift=-30, showarrow=False, font=dict(size=config.PLOT_NOTE_FONT_SIZE,
                                                              color=config.CHARCOAL))
    fig.update_layout(margin=dict(b=(fig.layout.margin.b or _MARGIN["b"]) + 24))


def _panel_px(height, rows):
    """Height of one panel's plot area in px (estimate from the figure height)."""
    return max((height - 150) / rows * 0.94, 60.0)


def _add_time_slider(fig, t, frames, readout=None, ticks=None, active=None, pad_t=44, notes=None):
    """Slider (one step per frame) and play/pause buttons on a Sand band under the plot, with a
    readout ("6.00 h / 12.00 h") right of it; the defaults are for time in hours.

    ``frames``: one (traces, trace indices) per step. ``notes(i)``: annotations that change with
    the step; each frame also carries the readout (the slider's own "current value" would show
    the step labels, which are empty except at a few round values). ``ticks``: step index ->
    label under the rail (default: the time axis' ticks, "0 h" ... "12 h"). ``pad_t``: space between the x axis and the slider (px). Call it last: the
    frames replace the figure's annotations.
    """
    t = np.asarray(t, dtype=float)
    n = len(frames)
    active = n - 1 if active is None else active
    if readout is None:
        def readout(v, end=float(t[-1])):
            return f"{v:.2f} h / {end:.2f} h"
    if ticks is None:
        axis = _time_axis(_time_range(t))
        ticks = {int(np.argmin(np.abs(t - v))): text for v, text in zip(axis["tickvals"], axis["ticktext"])}
    x0, length = 0.06, 0.74                       # slider position (paper)
    for i, text in ticks.items():                 # labels under the rail (Plotly would thin them out)
        frac = i / max(n - 1, 1)
        fig.add_annotation(text=text, x=x0 + length * frac, xref="paper", xshift=10 * (1 - 2 * frac),
                           y=0, yref="paper", yanchor="top", yshift=-(pad_t + 26), showarrow=False,
                           font=dict(size=config.PLOT_NOTE_FONT_SIZE, color=config.TEXT_2))
    fixed = list(fig.layout.annotations)
    mono = dict(family=config.PLOT_MONO_FONT_FAMILY, size=config.PLOT_READOUT_FONT_SIZE, color=config.CHARCOAL)

    own = [notes(i) if notes else [] for i in range(n)]
    most = max((len(o) for o in own), default=0)

    def annotations(i):
        # Same number in every frame: Plotly merges a frame's annotations into the figure's by
        # position, so a shorter list would leave old ones standing (spares are hidden).
        clock = go.layout.Annotation(text=readout(t[i]), x=1, xref="paper", xanchor="right", y=0, yref="paper",
                                     yanchor="middle", yshift=-(pad_t + 12), showarrow=False, font=mono)
        spare = [go.layout.Annotation(text="", visible=False, showarrow=False)] * (most - len(own[i]))
        return fixed + [clock] + [go.layout.Annotation(note) for note in own[i]] + spare

    fig.frames = [go.Frame(name=str(i), data=data, traces=idx, layout=dict(annotations=annotations(i)))
                  for i, (data, idx) in enumerate(frames)]
    fig.layout.annotations = annotations(active)
    steps = [dict(method="animate", label=str(i),
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for i in range(n)]
    band = dict(type="rect", xref="paper", yref="paper", ysizemode="pixel", yanchor=0, x0=0, x1=1,
                y0=-(pad_t - 10), y1=-(pad_t + 36), fillcolor=config.SAND, line_width=0, layer="below")
    fig.add_shape(band)
    fig.update_layout(
        sliders=[dict(active=active, steps=steps, x=x0, len=length, xanchor="left", y=0, yanchor="top",
                      pad=dict(t=pad_t, b=0, l=0, r=0), currentvalue=dict(visible=False),
                      ticklen=0, minorticklen=0,         # no row of step ticks under the rail
                      font=dict(color="rgba(0,0,0,0)"))],
        updatemenus=list(fig.layout.updatemenus or []) + [dict(type="buttons", direction="left", x=0.055, y=0, xanchor="right", yanchor="top",
                          pad=dict(t=pad_t - 4, r=4), showactive=False, name="play",
                          bgcolor=config.CHARCOAL, bordercolor=config.CHARCOAL,
                          font=dict(color="white", size=config.PLOT_LEGEND_FONT_SIZE), buttons=[
                              dict(label="▶", method="animate", args=[None, dict(
                                  frame=dict(duration=config.SLIDER_FRAME_MS, redraw=True),
                                  transition=dict(duration=0), fromcurrent=False)]),
                              dict(label="❚❚", method="animate", args=[[None], dict(
                                  frame=dict(duration=0, redraw=False), mode="immediate",
                                  transition=dict(duration=0))])])],
        margin=dict(b=pad_t + 60))


def _add_scale_buttons(fig, axes, log_active, labels=config.SCALE_BUTTON_LABELS, ys=None,
                       colour=None, layout=None):
    """Linear/Log buttons (top right, on the title row) that switch the scale of the plot.

    ``axes``: layout key ("yaxis", "yaxis2", ...) -> (linear settings, log settings).
    ``ys``: trace index -> (linear y, log y); the buttons also switch the data, so the log
    version can leave zeros out as gaps (and labels move with their lines). ``colour``: heatmap
    trace-0 attribute -> (linear, log) values, for a colour scale instead of axes. ``layout``:
    other layout attributes -> (linear, log). ``log_active`` marks the pressed button. Both
    buttons are drawn alike (showactive=False: Plotly's pressed-button colour is a fixed pale
    blue); the label shows the state, bold Charcoal when pressed, TEXT_2 otherwise (each button
    sets both labels).
    """
    menus = list(fig.layout.updatemenus or [])
    index = len(menus)

    def bold_labels(which):
        return {f"updatemenus[{index}].buttons[{n}].label": _button_label(labels[n], n == which)
                for n in (0, 1)}

    def args(which):
        extra = {key: pair[which] for key, pair in (layout or {}).items()}
        if colour is not None:
            return [{key: [pair[which]] for key, pair in colour.items()}, {**bold_labels(which), **extra}, [0]]
        lay = {f"{name}.{k}": v for name, pair in axes.items() for k, v in pair[which].items()}
        lay.update(bold_labels(which))
        lay.update(extra)
        if not ys:
            return [lay]
        idx = sorted(ys)
        return [{"y": [ys[i][which] for i in idx]}, lay, idx]

    method = "update" if colour is not None or ys else "relayout"
    active = 1 if log_active else 0
    buttons = [dict(label=_button_label(labels[n], n == active), method=method, args=args(n))
               for n in (0, 1)]
    menus.append(dict(type="buttons", direction="right", active=active, showactive=False, name="scale",
                      x=1.0, xanchor="right", y=1.0, yanchor="bottom", pad=dict(t=0, r=0, b=0),
                      buttons=buttons))
    fig.update_layout(updatemenus=menus)
    return fig


def _button_label(text, pressed):
    """Pressed: bold Charcoal; the other: TEXT_2 (the label is all a button can change)."""
    return f"<b>{text}</b>" if pressed else f'<span style="color:{config.TEXT_2}">{text}</span>'


def _message_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=config.TEXT_2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, "", 240)


# --- showing figures ------------------------------------------------------------------------------

def show_in_widget(*figs) -> None:
    """Show figures inside a widget's output area (the run panels' plots).

    On Colab, ``fig.show()`` inside a widget draws nothing: the figure's script runs before
    Colab has loaded plotly.js ("Plotly is not defined" in the browser console). There each
    figure goes into its own small page (an iframe), which loads plotly.js before drawing.
    Elsewhere this is ``fig.show()``, so local Jupyter keeps its built-in (offline) plotly.js.
    """
    if "google.colab" not in sys.modules:
        for fig in figs:
            fig.show()
        return
    from IPython.display import HTML, display
    for fig in figs:
        display(HTML(widget_iframe(fig)))


def widget_iframe(fig) -> str:
    """The figure as a self-contained page in an ``<iframe srcdoc>`` (used by show_in_widget).

    The page links the web fonts (config.FONT_CSS) and redraws the figure once they have
    loaded (theme.FONT_SCRIPT), which also draws the card frame (1 px config.LINE border,
    rounded corners). Wrapped in a ``<div>``: IPython warns about HTML that starts with
    ``<iframe``."""
    page = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"responsive": True},
                       post_script=theme.FONT_SCRIPT)
    page = page.replace("<head>", f"<head>{config.FONT_CSS}<style>html, body {{ margin: 0; }}</style>", 1)
    height = int(fig.layout.height or config.PLOT_HEIGHT) + 4
    return (f'<div><iframe srcdoc="{html.escape(page, quote=True)}" '
            f'style="width: 100%; height: {height}px; border: 0;"></iframe></div>')


def style(fig, title, height=None, legend_right=False, subtitle=None) -> go.Figure:
    """Give any Plotly figure the look of these plots: the "reflective" template (theme.py),
    title, legend and margins.

    The legend goes under the title (and subtitle), or to the right of the plot with
    ``legend_right=True``. ``subtitle``: a second, smaller line under the title; say there what
    is plotted and in which units, e.g. "SOA (µg/m³) vs time (h)" (Plotly titles do not wrap, so
    put details there rather than make the title wider than a notebook cell). Traces keep
    colours and dashes they set themselves; others take config.SERIES_COLORS and
    config.SERIES_DASHES in order, so lines differ in dash as well as colour. A figure that
    sets a dash on any line already uses dash for something (e.g. "solid: model, dashed:
    formula"): there lines without a dash stay solid. Subplot titles
    (``make_subplots(subplot_titles=...)``) get the title font, SemiBold, a size smaller.
    """
    height = height or config.PLOT_HEIGHT
    subtitle = subtitle or fig.layout.title.subtitle.text        # or one set before style()
    panel_titles = [note for note in fig.layout.annotations if _is_subplot_title(note)]
    top = config.PLOT_TITLE_TOP + (20 if title else 0)
    if subtitle:
        top += config.PLOT_SUBTITLE_SPACE + (0 if title else 4)
    legend_top = top + 6
    rows = 0 if legend_right else _legend_rows(fig)
    if legend_right:
        legend = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, xref="paper", yref="paper")
    else:
        legend = dict(orientation="h", yref="container", yanchor="top", y=1 - legend_top / height,
                      xref="container", xanchor="left", x=0, indentation=6)
        top = legend_top + rows * config.PLOT_LEGEND_ROW
    top += 12 + (22 if panel_titles else 0)
    if any(note.name == _NIGHT_ABOVE for note in fig.layout.annotations):
        top += _NIGHT_LABEL_PX[1] + 2            # room for "night" just above the plot
    if title and subtitle:
        top = max(top, 88)          # Plotly moves a title up that would come closer to the plot
    scale_menu = any(menu.name == "scale" for menu in fig.layout.updatemenus)
    axis_titles = any(getattr(fig.layout[k], "title", None) is not None and fig.layout[k].title.text
                      for k in fig.layout if k.startswith("yaxis"))
    margin = dict(l=_MARGIN["l"] + (36 if axis_titles else 0), r=fig.layout.margin.r or _MARGIN["r"], t=top,
                  b=fig.layout.margin.b or _MARGIN["b"])
    if scale_menu:
        # Plotly would grow the top margin for the Linear/Log buttons above the plot (and the
        # buttons would end up under the legend): fixed margins put them on the title row.
        margin["autoexpand"] = False
    fig.update_layout(
        template=config.PLOT_TEMPLATE, height=height, title=dict(text=_title_text(title, subtitle),
                                                                 subtitle=dict(text=None)),
        legend=legend, margin=margin)
    # Linear/Log on the title row, top right: at the card's edge (as on figures with the default
    # right margin) also where labels at the line ends widen the margin. Menus are placed in
    # plot-area units, so this assumes the usual plot width (_PLOT_WIDTH_PX, an estimate).
    menu_x = 1 + max(margin["r"] - _MARGIN["r"], 0) / _PLOT_WIDTH_PX
    for menu in fig.layout.updatemenus:
        if menu.name == "scale":
            menu.update(x=menu_x, xanchor="right", y=1, yanchor="bottom",
                        pad=dict(t=0, r=0, b=max(top - config.PLOT_TITLE_TOP + 2 - 26, 0)))
    for note in panel_titles:
        note.font.update(family=config.PLOT_TITLE_FONT_FAMILY, size=config.PLOT_PANEL_TITLE_FONT_SIZE,
                         color=config.CHARCOAL, weight=config.PLOT_TITLE_FONT_WEIGHT)
    lines = [trace for trace in fig.data if trace.type == "scatter" and trace.mode != "text"]
    if any(trace.line.dash is not None for trace in lines):
        for trace in lines:
            if trace.line.dash is None:
                trace.line.dash = "solid"          # not the template's dash cycle
    return fig


def _title_text(title, subtitle):
    """Title and, on a second line, the subtitle (smaller, TEXT_2). One text, so the subtitle
    sits right under the title, as in the app (Plotly's own subtitle keeps a wider gap and moves
    with the top margin)."""
    if not subtitle:
        return title
    sub = (f'<span style="font-size:{config.PLOT_SUBTITLE_FONT_SIZE}px;font-weight:400;'
           f'color:{config.TEXT_2}">{subtitle}</span>')
    return f"{title}<br>{sub}" if title else sub


def _subtitle_of(fig) -> str | None:
    """The subtitle style() wrote into a figure's title (None if it has none)."""
    found = re.search(r'<span style="font-size:[^"]*;font-weight:400;[^"]*">(.*)</span>$',
                      fig.layout.title.text or "")
    return found.group(1) if found else None


def _title_of(fig) -> str:
    """The title of a figure from style(), without its subtitle."""
    return re.sub(r'(<br>)?<span style="font-size:[^"]*;font-weight:400;[^"]*">.*</span>$', "",
                  fig.layout.title.text or "")


def _legend_rows(fig):
    """Rows the horizontal legend needs (estimate at config.PLOT_LEGEND_WIDTH_ESTIMATE px)."""
    entries = [trace for trace in fig.data if trace.showlegend is not False and trace.name
               and not (trace.type == "scatter" and trace.mode == "text") and trace.type != "heatmap"]
    if len(entries) < 2 and not any(trace.showlegend for trace in entries):
        return 0
    # A horizontal legend in "grouped" order (Plotly's default once traces have legend groups)
    # stacks each group's entries vertically: a column per group, as tall as its entries.
    order = fig.layout.legend.traceorder or ("grouped" if any(t.legendgroup for t in entries) else "normal")
    columns = {}
    for n, trace in enumerate(entries):
        key = trace.legendgroup if "grouped" in order and trace.legendgroup else f"\0{n}"
        columns.setdefault(key, []).append(30 + 5 + _CHAR_PX * 0.95 * len(_plain(trace.name)) + 14)
    rows, line_rows, used = 0, 0, 0.0
    for widths in columns.values():
        width = max(widths)
        if used and used + width > config.PLOT_LEGEND_WIDTH_ESTIMATE:
            rows, line_rows, used = rows + line_rows, 0, 0.0
        used += width
        line_rows = max(line_rows, len(widths))
    return rows + line_rows


def _is_subplot_title(note) -> bool:
    """True for the annotations make_subplots() writes for subplot_titles (Plotly's own
    settings: paper coordinates, no arrow, anchored at the bottom, font size 16)."""
    return (note.xref == "paper" and note.yref == "paper" and note.showarrow is False
            and note.yanchor == "bottom" and note.font is not None and note.font.size == 16)
