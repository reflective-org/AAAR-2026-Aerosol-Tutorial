import re

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

import aaar_tutorial as at
from aaar_tutorial import config, plots

SINGLE_RUN_PLOTS = [
    lambda r: plots.size_distribution(r),
    lambda r: plots.size_distribution(r, quantity="mass", log_y=True),
    lambda r: plots.banana(r),
    lambda r: plots.banana(r, quantity="mass"),
    lambda r: plots.totals(r),
    lambda r: plots.time_series(r, ["cs_per_s", "h2so4_cm3"], log_y=True),
    lambda r: plots.composition(r),
    lambda r: plots.composition(r, fraction=True),
    lambda r: plots.volatility(r),
    lambda r: plots.gases(r),
    lambda r: plots.radicals(r),
    lambda r: plots.photolysis(r),
    lambda r: plots.top_contributors(r),
]


@pytest.fixture(scope="module", params=config.PRESETS)
def example(request):
    return at.load_example(request.param)


@pytest.mark.parametrize("make", SINGLE_RUN_PLOTS)
def test_every_plot_builds(example, make):
    fig = make(example)
    assert isinstance(fig, go.Figure)
    fig.to_json()                      # validates the whole figure


@pytest.mark.parametrize("make", [plots.size_distribution, plots.volatility])
def test_slider_has_one_frame_per_output(example, make):
    fig = make(example)
    assert len(fig.frames) == len(example.time_h)
    assert len(fig.layout.sliders[0].steps) == len(example.time_h)


def test_level1_has_no_chemistry_plots():
    r = at.load_example("level1_volatility")
    for fig in (plots.gases(r), plots.radicals(r), plots.photolysis(r), plots.top_contributors(r)):
        assert "Level 1" in fig.layout.annotations[0].text


def _lines(fig):
    """The data lines (not the labels at their ends)."""
    return [t for t in fig.data if t.type == "scatter" and t.mode != "text"]


def _end_labels(fig):
    return [t for t in fig.data if t.type == "scatter" and t.mode == "text"]


def test_level3_night_shading():
    fig = plots.gases(at.load_example("level3_day_night"))
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].layer == "below"
    assert fig.layout.shapes[0].fillcolor == config.NIGHT_COLOR
    assert [a.text for a in fig.layout.annotations] == [config.NIGHT_LABEL]
    assert "SZA > 90°" in plots._subtitle_of(fig)                  # no legend entry for night
    assert not any("night" in (t.name or "") for t in fig.data)


def test_gases_default_visibility():
    fig = plots.gases(at.load_example("level2_limonene_soa"))
    shown = {t.name for t in fig.data if t.visible is True}
    assert shown == {"Limonene", "O₃", "NO", "NO₂"}


def test_comparisons():
    a = at.load_example("level3_day_night")
    b = at.load_example("level2_limonene_soa")
    fig = plots.totals([a, b], labels=["A", "B"])
    assert {t.name for t in fig.data if t.showlegend} >= {"A", "B"}
    run_a, run_b = (t for t in _lines(fig) if t.showlegend)
    assert (run_a.line.color, run_a.line.dash) == (config.RUN_A_COLOR, "solid")
    assert (run_b.line.color, run_b.line.dash) == (config.RUN_B_COLOR, config.RUN_B_DASH)
    assert {t.text[0].strip() for t in _end_labels(fig)} == {"A", "B"}
    assert all(t.text[0].startswith("\u2002") for t in _end_labels(fig))   # a gap after the line end
    fig = plots.compare_size_distributions([a, b], labels=["A", "B"], time_h=3)
    assert [t.name for t in _lines(fig)] == ["A: initial", "A", "B: initial", "B"]   # different seeds
    assert [t.text[0] for t in _end_labels(fig)] == ["A", "B"]                      # at the peaks
    assert "t = 3 h" in plots._title_of(fig)
    assert any(a.text.startswith("Mode ") for a in fig.layout.annotations)         # caption
    fig.to_json()
    with pytest.raises(ValueError, match="labels"):
        plots.totals([a, b], labels=["only one"])


def test_same_start_gets_one_initial_curve():
    r = at.load_example("level1_volatility")
    fig = plots.compare_size_distributions([r, r], labels=["A", "B"])
    assert [t.name for t in _lines(fig)] == ["initial", "A", "B"]


def test_log_axis_labels_and_gaps():
    r = at.load_example("level3_day_night")
    fig = plots.time_series(r, ["OH", "N_cm3"], log_y=True)
    oh = fig.data[0].y
    assert oh[0] is None or oh[0] != oh[0]                  # t = 0 radical is a gap, not 0
    ticks = fig.layout.yaxis.ticktext
    assert all("<sup>" in t or t.replace(",", "").isdigit() for t in ticks)
    assert "<br>" in fig.layout.yaxis.title.text            # several panels: unit on its own line


def test_n2o5_is_shown_in_ppt():
    r = at.load_example("level3_day_night")
    fig = plots.time_series(r, ["N2O5"])
    assert fig.data[0].y[-1] == pytest.approx(1e3 * at.series(r, "N2O5")[-1])
    assert "N₂O₅ (ppt)" in plots._subtitle_of(fig)
    assert not fig.layout.yaxis.title.text                  # one panel: the unit is in the subtitle


def test_new_series_keys_plot():
    r = at.load_example("level1_volatility")
    fig = plots.time_series([r, r], ["SOA_ugm3", "org_gas_ugm3", "gmd_nm"], labels=["a", "b"])
    assert plots._title_of(fig) == "SOA, organic vapor and geometric mean diameter"   # sentence case
    assert plots._join(["Condensation sink", "Gas-phase H₂SO₄", "N₂O₅", "J(NO₂)"]) == \
        "Condensation sink, gas-phase H₂SO₄, N₂O₅ and J(NO₂)"


def test_sunlight_figure():
    import numpy as np
    t = np.arange(0, 24.01, 0.5)
    day = {"label": "test", "local_start_h": 12, "time_h": t, "sza_deg": 90 + 60 * np.sin(t / 24 * 2 * np.pi),
           "J_NO2": np.clip(np.cos(t / 24 * 2 * np.pi), 0, None) * 8e-3}
    fig = plots.sunlight([day])
    assert len(fig.data) == 2 and fig.layout.xaxis2.ticktext[0] == "12:00"


def test_style_is_public():
    import plotly.graph_objects as go
    fig = plots.style(go.Figure(), "my figure", legend_right=True)
    assert fig.layout.title.text == "my figure" and fig.layout.legend.x == 1.02
    assert plots._subtitle_of(fig) is None
    sub = plots.style(go.Figure(), "my figure", subtitle="details")
    assert plots._subtitle_of(sub) == "details" and plots._title_of(sub) == "my figure"
    assert sub.layout.margin.t > fig.layout.margin.t
    early = go.Figure(layout=dict(title=dict(subtitle=dict(text="set before"))))
    assert plots._subtitle_of(plots.style(early, "t")) == "set before"


def test_long_titles_put_details_in_the_subtitle():
    fig = plots.top_contributors(at.load_example("level2_limonene_soa"))
    assert "N-containing" in plots._subtitle_of(fig) and "N-containing" not in plots._title_of(fig)


def test_same_name_runs_get_numbered_labels():
    r = at.load_example("level1_volatility")
    fig = plots.totals([r, r])
    names = [_plain(t.name) for t in fig.data if t.showlegend]          # "A  run 1", "B  run 2"
    assert names == ["A  run 1", "B  run 2"]


def test_partitioning_explainer():
    fig = plots.partitioning(highlight_cstar=1.0, highlight_label="surrogate")
    widths = list(fig.data[0].marker.line.width)
    assert widths[list(fig.data[0].x).index(0)] > 0 and sum(w > 0 for w in widths) == 1
    assert any("surrogate" in t for t in fig.layout.xaxis.ticktext)
    assert len(fig.frames) == len(config.PARTITIONING_COA_UGM3)
    start = config.PARTITIONING_COA_UGM3.index(config.PARTITIONING_START_COA_UGM3)
    assert fig.layout.sliders[0].active == start
    x = list(fig.data[0].x)
    halves = 0
    for frame, coa in zip(fig.frames, config.PARTITIONING_COA_UGM3):
        particle, gas = (np.asarray(trace.y) for trace in frame.data[:2])
        np.testing.assert_allclose(particle + gas, 100.0)
        k = round(float(np.log10(coa)))
        if np.isclose(10.0 ** k, coa) and k in x:                # c* = C_OA is split half and half
            assert particle[x.index(k)] == pytest.approx(50.0)
            halves += 1
        assert frame.data[2].x[0] == pytest.approx(np.log10(coa))
    assert halves >= 3
    assert min(config.PARTITIONING_COA_UGM3) == 0.1 and max(config.PARTITIONING_COA_UGM3) == 1000
    assert np.all(np.diff(np.asarray(fig.frames[0].data[0].y)) < 0)   # less volatile: more in particles


def test_compare_runs_on_different_size_grids():
    coarse = at.load_example("level1_volatility")                  # 40 bins
    fine = at.run("level1_volatility", duration_h=0.5, numerics={"nbins": 80}, progress=False, name="80 bins")
    assert len(coarse.diameter_nm) != len(fine.diameter_nm)
    fig = plots.compare_size_distributions([coarse, fine])
    assert len(_lines(fig)) == 4                    # one initial and one final curve per run (different grids)
    lo, hi = fig.layout.xaxis.range
    assert lo < np.log10(60) < hi


def _plain(label):
    return re.sub(r"<[^>]+>", "", label)


def _labels(menu):
    return [_plain(b.label) for b in menu.buttons]


def _scale_menu(fig):
    menus = [m for m in fig.layout.updatemenus if m.buttons and _plain(m.buttons[-1].label).startswith("Log")]
    assert len(menus) == 1
    return menus[0]


def test_banana_white_floor_and_colour_buttons():
    r = at.load_example("level1_volatility")
    fig = plots.banana(r)
    assert fig.data[0].colorscale[0][1].lower() == "#ffffff"
    assert not np.isnan(np.asarray(fig.data[0].z, dtype=float)).any()      # empty cells drawn white
    menu = _scale_menu(fig)
    assert _labels(menu) == ["Linear colour", "Log colour"] and menu.active == 1
    lin, log = (b.args[0] for b in menu.buttons)
    assert lin["zmin"] == [0.0]
    assert log["zmax"][0] == pytest.approx(np.log10(np.nanmax(np.asarray(r.dndlog10dp_cm3, dtype=float))))


@pytest.mark.parametrize("make, n_axes, log_first", [
    (lambda r: plots.size_distribution(r), 1, False),
    (lambda r: plots.compare_size_distributions([r, r], labels=["a", "b"]), 1, False),
    (lambda r: plots.totals(r), 5, False),
    (lambda r: plots.gases(r), 1, True),
    (lambda r: plots.radicals(r), 1, True),
])
def test_linear_log_buttons(make, n_axes, log_first):
    r = at.load_example("level3_day_night")
    menu = _scale_menu(make(r))
    assert _labels(menu) == ["Linear", "Log"] and menu.active == int(log_first)
    layouts = [b.args[0] if b.method == "relayout" else b.args[1] for b in menu.buttons]
    assert sum(k.endswith(".type") for k in layouts[1]) == n_axes
    assert all(v == "log" for k, v in layouts[1].items() if k.endswith(".type"))
    assert all(v == "linear" for k, v in layouts[0].items() if k.endswith(".type"))


def test_log_button_turns_zeros_into_gaps():
    r = at.load_example("level3_day_night")
    menu = _scale_menu(plots.radicals(r))
    lin_y, log_y = (b.args[0]["y"] for b in menu.buttons)
    assert lin_y[0][0] == 0.0 and np.isnan(log_y[0][0])                    # OH at t = 0


def test_widget_iframe_is_a_self_contained_page():
    import html as html_mod
    import re
    fig = plots.time_series(at.load_example("level1_volatility"), ["SOA_ugm3"])
    frame = plots.widget_iframe(fig)
    assert frame.startswith('<div><iframe srcdoc="') and frame.endswith("></iframe></div>")
    page = html_mod.unescape(re.search(r'srcdoc="([^"]*)"', frame).group(1))
    # plotly.js is loaded by a script tag before the figure's own script, in one document
    assert page.index('src="https://cdn.plot.ly/plotly-') < page.index("Plotly.newPlot")
    assert '"' not in re.search(r'srcdoc="([^"]*)"', frame).group(1)
    assert f"height: {int(fig.layout.height) + 4}px" in frame


def test_show_in_widget_uses_the_iframe_only_on_colab(monkeypatch):
    import sys
    import types
    shown, displayed = [], []
    fig = plots.time_series(at.load_example("level1_volatility"), ["SOA_ugm3"])
    monkeypatch.setattr(type(fig), "show", lambda self, *a, **k: shown.append(self))
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", lambda obj: displayed.append(obj))
    plots.show_in_widget(fig)
    assert shown == [fig] and displayed == []
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    plots.show_in_widget(fig, fig)
    assert len(shown) == 1 and len(displayed) == 2
    assert displayed[0].data.startswith("<div><iframe srcdoc=")



def test_reflective_template_is_registered_and_applied():
    assert config.PLOT_TEMPLATE == "reflective" and "reflective" in pio.templates
    template = pio.templates[config.PLOT_TEMPLATE]
    assert template.layout.font.color == config.CHARCOAL
    title = template.layout.title.font
    assert "Open Sans" in template.layout.font.family and "Open Sans" in title.family   # no serif titles
    assert title.weight == 600 and title.color == config.CHARCOAL
    assert template.layout.xaxis.tickfont.color == config.TEXT_2                      # secondary text
    assert list(template.layout.colorway) == list(config.SERIES_COLORS)
    axis = template.layout.xaxis
    assert axis.showline and not axis.mirror and axis.linecolor == config.SLATE and axis.linewidth == 1
    assert axis.ticks == "outside" and axis.ticklen == 3 and axis.showgrid is False
    assert template.layout.legend.borderwidth == 0
    assert template.layout.hoverlabel.bordercolor == config.LINE
    assert all(t.line.width == config.PLOT_LINE_WIDTH for t in template.data.scatter)
    fig = plots.style(go.Figure(go.Scatter(y=[1, 2])), "mine")
    assert fig.layout.template.layout.font.family == template.layout.font.family
    for fig in (plots.totals(at.load_example("level1_volatility")), plots.partitioning()):
        assert fig.layout.template.layout.title.font.family == config.PLOT_TITLE_FONT_FAMILY
    assert pio.templates.default != config.PLOT_TEMPLATE        # Plotly's global default is untouched


def test_series_differ_in_colour_dash_and_marker():
    n = len(config.SERIES_COLORS)
    assert len(config.SERIES_DASHES) == n and len(config.SERIES_MARKERS) == n + 1   # cycles of different length
    assert len(set(config.SERIES_COLORS)) == len(set(config.SERIES_DASHES)) == n
    assert config.SERIES_COLORS[1] != config.PARTICLE_COLOR      # "contains N" vs "no N" bars
    r = at.load_example("level3_day_night")
    fig = plots.time_series([r, r, r], ["N_cm3"], labels=["a", "b", "c"])
    lines = [t for t in _lines(fig) if t.name in ("a", "b", "c")]
    assert len({t.line.color for t in lines}) == len({t.line.dash for t in lines}) == 3
    assert all(t.mode == "lines" for t in lines)                              # no markers, as in the app
    assert [t.text[0].strip() for t in _end_labels(fig)] == ["a", "b", "c"]   # named at the line ends
    assert list(fig.layout.xaxis.range) == [r.time_h[0], r.time_h[-1]]
    assert plots._style(0, markers=True)["mode"] == "lines+markers"           # opt-in for own figures


def test_scale_buttons_mark_the_pressed_one():
    r = at.load_example("level3_day_night")
    fig = plots.size_distribution(r)                       # play/pause menu first, then Linear/Log
    menu = _scale_menu(fig)
    assert menu.showactive is False                         # no pale-blue pressed button
    assert all(m.showactive is False for m in fig.layout.updatemenus)
    index = list(fig.layout.updatemenus).index(menu)
    assert menu.buttons[0].label == "<b>Linear</b>" and "<b>" not in menu.buttons[1].label
    to_log = menu.buttons[1].args[0]
    assert to_log[f"updatemenus[{index}].buttons[1].label"] == "<b>Log</b>"
    assert "<b>" not in to_log[f"updatemenus[{index}].buttons[0].label"]


@pytest.mark.parametrize("make", [plots.gases, plots.radicals])
def test_every_species_line_has_its_own_style(example, make):
    fig = make(example)
    if not example.gases_ppb:
        return                                              # Level 1: message figure
    lines = _lines(fig)
    pairs = [(t.line.color, t.line.dash) for t in lines]
    assert len(set(pairs)) == len(pairs), pairs
    # each line has a label at its end in its legend group (a legend click hides both)
    labels = {t.legendgroup: t for t in _end_labels(fig)}
    assert {t.legendgroup for t in lines} == set(labels)
    assert all(labels[t.legendgroup].visible == t.visible for t in lines)
    assert all(labels[t.legendgroup].showlegend is False for t in lines)


def test_species_follow_the_app_styles():
    r = at.load_example("level3_day_night")                 # NO2 is held in Level 3
    lines = {t.legendgroup: t for t in _lines(plots.gases(r))}
    assert (lines["NO2"].line.color, lines["NO2"].line.dash) == config.HELD_STYLE
    assert lines["NO2"].name == "NO₂ (held)"
    assert (lines["N2O5"].line.color, lines["N2O5"].line.dash) == config.SPECIES_STYLE["N2O5"]
    lines = {t.legendgroup: t for t in _lines(plots.gases(at.load_example("level2_limonene_soa")))}
    assert (lines["NO2"].line.color, lines["NO2"].line.dash) == config.SPECIES_STYLE["NO2"]
    labels = {t.legendgroup: t for t in _end_labels(plots.radicals(r))}
    assert labels["HO2"].textfont.color == config.GREEN_INK                  # Green is not for text


def test_single_run_totals_use_the_composition_colours():
    r = at.load_example("level3_day_night")
    fig = plots.totals(r)
    by_panel = {t.yaxis: t.line.color for t in fig.data if t.mode == "lines"}
    colors = dict(zip(config.DEFAULT_TOTALS, (by_panel[f"y{i if i > 1 else ''}"]
                                              for i in range(1, len(config.DEFAULT_TOTALS) + 1))))
    layers = {key: color for key, _, color in config.COMPOSITION_LAYERS}
    for key in ("SOA_ugm3", "SO4_ugm3", "NIT_ugm3"):
        assert colors[key] == layers[key]
    assert colors["N_cm3"] == config.CHARCOAL
    comp = {t.legendgroup: t.fillcolor for t in plots.composition(r).data if t.fillcolor}
    assert comp["SOA"] == plots._rgba(colors["SOA_ugm3"], config.COMPOSITION_FILL_OPACITY)
    assert comp["Sulfate"] == plots._rgba(colors["SO4_ugm3"], config.COMPOSITION_FILL_OPACITY)
    assert all(t.line.dash == "solid" for t in fig.data if t.mode == "lines")


def test_template_cycles_dashes_and_internal_lines_stay_solid():
    template = pio.templates[config.PLOT_TEMPLATE]
    assert [t.line.dash for t in template.data.scatter] == list(config.SERIES_DASHES)
    r = at.load_example("level3_day_night")
    fig = plots.size_distribution(r)
    assert all(t.line.dash in ("solid", "dot") for t in fig.data)
    for fig in (plots.composition(r), plots.photolysis(r)):
        assert all(t.line.dash is not None for t in _lines(fig))


def test_subplot_titles_use_the_title_font():
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("A", "B"))
    fig.add_annotation(text="note", x=1, y=1, showarrow=True)
    plots.style(fig, "panels")
    titles = [a for a in fig.layout.annotations if a.text in ("A", "B")]
    assert all(a.font.family == config.PLOT_TITLE_FONT_FAMILY for a in titles)
    assert all(a.font.size == config.PLOT_PANEL_TITLE_FONT_SIZE for a in titles)
    note = next(a for a in fig.layout.annotations if a.text == "note")
    assert note.font.family is None


def test_web_fonts_are_shipped_with_the_figures():
    import html as html_mod
    from aaar_tutorial import theme
    fig = plots.totals(at.load_example("level1_volatility"))
    page = html_mod.unescape(re.search(r'srcdoc="([^"]*)"', plots.widget_iframe(fig)).group(1))
    head = page[:page.index("</head>")]
    assert all(url.replace("&", "&amp;") in head for url in config.FONT_CSS_URLS)
    assert "__reflectiveFonts" in page and "{plot_id}" not in page
    for name in ("notebook", "notebook_connected", "colab", "iframe", "browser"):
        scripts = pio.renderers[name].post_script
        assert sum("__reflectiveFonts" in s for s in scripts) == 1
    theme.register()                                        # registering again adds no copy
    assert sum("__reflectiveFonts" in s for s in pio.renderers["colab"].post_script) == 1
    assert any("JetBrains+Mono" in url for url in config.FONT_CSS_URLS)
    # the card frame: only on figures in the template (they start with its font family)
    assert f"1px solid {config.LINE}" in theme.FONT_SCRIPT and config.PLOT_FONT_FAMILY in theme.FONT_SCRIPT


def test_composition_night_bands_show_through_the_stack():
    r = at.load_example("level3_day_night")
    assert all(sh.layer == "below" for sh in plots.composition(r).layout.shapes)
    fig = plots.composition(r, fraction=True)        # the 100 % stack fills the plot: band over it
    assert fig.layout.shapes and all(sh.layer == "above" and sh.fillcolor == config.NIGHT_OVERLAY_COLOR
                                     for sh in fig.layout.shapes)
    night = [a for a in fig.layout.annotations if a.text == config.NIGHT_LABEL]
    assert night and all(a.yanchor == "bottom" and a.y == 1 for a in night)   # above the plot
    assert all(t.fillcolor.endswith(f"{config.COMPOSITION_FILL_OPACITY:g})") for t in fig.data)
    assert list(fig.layout.yaxis.range) == [0, 100]
    boxes = [a for a in fig.layout.annotations if a.bgcolor == "white"]      # label at each band
    assert len(boxes) == len(fig.data) and all("■" in a.text for a in boxes)
    assert fig.data[-1].name.endswith("% at 24 h")                         # values in the legend
    one = plots.composition(at.load_example("level1_volatility"))            # zero layers left out
    assert [t.legendgroup for t in one.data] == ["Sulfate", "POA", "SOA"]


def test_log_size_distribution_ticks_match_the_other_axes():
    r = at.load_example("level2_limonene_soa")
    fig = plots.size_distribution(r, quantity="mass", log_y=True)
    assert fig.layout.yaxis.type == "log" and fig.layout.yaxis.ticktext
    assert fig.layout.yaxis.exponentformat is None
    log = _scale_menu(fig).buttons[1].args[0]
    assert list(log["yaxis.ticktext"]) == list(fig.layout.yaxis.ticktext)
    assert fig.layout.sliders[0].ticklen == 0


def test_custom_figures_cycle_dashes_unless_they_use_dashes_themselves():
    plain = plots.style(go.Figure([go.Scatter(y=[1, 2]), go.Scatter(y=[2, 3])]), "plain")
    assert [t.line.dash for t in plain.data] == [None, None]            # template cycle applies
    coded = plots.style(go.Figure([go.Scatter(y=[1, 2], line=dict(color="red")),
                                   go.Scatter(y=[2, 3], line=dict(color="red", dash="dash")),
                                   go.Scatter(y=[3, 4])]), "solid: model, dashed: formula")
    assert [t.line.dash for t in coded.data] == ["solid", "dash", "solid"]


def test_top_legend_clears_the_subplot_titles():
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("A", "B"))
    fig.add_trace(go.Scatter(y=[1, 2], name="a"))
    fig.add_trace(go.Scatter(y=[2, 1], name="b"))
    plain = make_subplots(rows=1, cols=2)
    plain.add_trace(go.Scatter(y=[1, 2], name="a"))
    plain.add_trace(go.Scatter(y=[2, 1], name="b"))
    plots.style(fig, "panels")
    plots.style(plain, "panels")
    assert fig.layout.legend.yref == "container" and fig.layout.legend.yanchor == "top"
    assert fig.layout.margin.t > plain.layout.margin.t


def test_tick_labels_follow_the_app():
    label = plots._tick_label
    assert [label(v) for v in (0, 2000, 4000, 0.004, 1.5)] == ["0", "2,000", "4,000", "0.004", "1.5"]
    assert label(6e4) == "6×10<sup>4</sup>" and label(1e5) == "10<sup>5</sup>"
    assert label(1e-4) == "10<sup>-4</sup>"
    assert label(0.01, log=True) == "10<sup>-2</sup>" and label(100, log=True) == "100"
    axis = plots._log_axis([6e4, 1.5e9])                              # every 2nd decade
    assert axis["ticktext"] == ["10<sup>4</sup>", "10<sup>6</sup>", "10<sup>8</sup>", "10<sup>10</sup>"]
    assert axis["range"] == [4, 10] and axis["autorange"] is False     # a tick beyond the data, both ends


def test_axes_carry_the_units_on_their_ticks():
    r = at.load_example("level1_volatility")
    fig = plots.size_distribution(r)
    x = fig.layout.xaxis
    assert x.ticktext[0].endswith(" nm") and all(np.log10(v) % 1 == 0 for v in x.tickvals)   # decades
    assert list(x.range) == [np.log10(x.tickvals[0]), np.log10(x.tickvals[-1])]
    assert not x.title.text and not fig.layout.yaxis.title.text           # units in the subtitle
    assert "(cm⁻³)" in plots._subtitle_of(fig)
    t = plots.banana(r).layout.xaxis
    assert t.ticktext[0] == "0 h" and t.ticktext[-1] == "12 h"
    assert all(unit == "µg/m³" for key, (_, unit) in config.SERIES_LABELS.items() if key.endswith("_ugm3"))


def test_scale_buttons_move_the_end_labels():
    r = at.load_example("level3_day_night")
    fig = plots.radicals(r)
    menu = _scale_menu(fig)
    labels = [i for i, t in enumerate(fig.data) if t.mode == "text"]
    lin, log = (b.args for b in menu.buttons)
    assert set(labels) <= set(lin[2]) and lin[2] == log[2]
    k = lin[2].index(labels[0])
    assert lin[0]["y"][k] != log[0]["y"][k]
    assert fig.layout.margin.autoexpand is False             # buttons on the title row, top right
    assert menu.yanchor == "bottom" and menu.pad.b > 0


def test_slider_frames_carry_the_readout():
    r = at.load_example("level1_volatility")
    fig = plots.volatility(r)
    counts = {len(frame.layout.annotations) for frame in fig.frames}
    assert counts == {len(fig.layout.annotations)}            # same length: no stale labels
    texts = [a.text for a in fig.frames[4].layout.annotations]
    assert f"{r.time_h[4]:.2f} h / {r.time_h[-1]:.2f} h" in texts
    assert any(t.startswith("C_OA = ") for t in texts)
    slider = fig.layout.sliders[0]
    assert slider.currentvalue.visible is False
    assert {a.text for a in fig.layout.annotations} >= {"0 h", "12 h"}  # tick labels under the rail


def test_banana_small_colour_bar_with_end_labels():
    fig = plots.banana(at.load_example("level1_volatility"))
    bar = fig.data[0].colorbar
    assert bar.orientation == "h" and bar.showticklabels is False and bar.yref == "container"
    menu = _scale_menu(fig)
    lin, log = (b.args[1] for b in menu.buttons)
    assert lin["annotations[0].text"] == "0" and lin["annotations[1].text"].endswith("cm⁻³")
    assert [a.text for a in fig.layout.annotations] == [log["annotations[0].text"], log["annotations[1].text"]]
    assert fig.layout.xaxis.mirror is not True and fig.layout.yaxis.mirror is not True   # no box


def test_card_frame_in_the_widget_page():
    import html as html_mod
    page = html_mod.unescape(re.search(r'srcdoc="([^"]*)"', plots.widget_iframe(
        plots.totals(at.load_example("level1_volatility")))).group(1))
    assert "reflectiveCard" in page


def test_log_axes_end_on_ticks():
    for values in ([40, 44], [0.008, 44], [5e-4, 3e-3], [0.0062, 0.0185], [6e4, 2.3e7], [0.04, 80]):
        axis = plots._log_axis(values)
        lo, hi = 10.0 ** np.array(axis["range"])
        # ends on the data tick itself when a value sits on a tick (10**log10(40) can come back
        # as 40.000000000000014, so compare to one part in 1e12)
        assert lo <= min(values) * (1 + 1e-12) and hi >= max(values) * (1 - 1e-12)
        assert lo == pytest.approx(axis["tickvals"][0]) and hi == pytest.approx(axis["tickvals"][-1])
        assert 2 <= len(axis["tickvals"]) <= config.LOG_AXIS_MAX_TICKS
    r = at.load_example("level3_day_night")
    log = _scale_menu(plots.radicals(r)).buttons[1].args[1]
    assert log["yaxis.range"] == [4, 10] and log["yaxis.autorange"] is False
    gases = plots.gases(at.load_example("level2_limonene_soa"))   # hidden CO (120 ppb) stays in view
    assert 10.0 ** gases.layout.yaxis.range[1] >= 120


def test_label_margins_only_with_labels():
    r = at.load_example("level1_volatility")
    one = plots.totals(r)
    assert not _end_labels(one) and one.layout.margin.r == plots._MARGIN["r"]
    assert _scale_menu(one).x == 1
    three = plots.totals([r, r, r], labels=["Level 1", "Level 2", "Level 3"])
    assert three.layout.margin.r > plots._MARGIN["r"] and _scale_menu(three).x > 1   # menu at the card edge


def test_legend_rows_count_legend_groups():
    a, b = at.load_example("level3_day_night"), at.load_example("level2_limonene_soa")
    fig = plots.compare_size_distributions([a, b], labels=["Level 3", "Level 2"], time_h=3)
    assert plots._legend_rows(fig) == 2                    # "initial" above each run, per group
    same = plots.compare_size_distributions([a, a], labels=["A", "B"])
    assert plots._legend_rows(same) == 1


def test_identical_peaks_get_labels_side_by_side():
    r = at.load_example("level1_volatility")
    fig = plots.compare_size_distributions([r, r], labels=["one", "two"])
    a, b = _end_labels(fig)
    assert (a.textposition, b.textposition) == ("top left", "top right") and a.y == b.y
    peak = float(np.nanmax(np.asarray(r.dndlog10dp_cm3, dtype=float)[-1]))
    assert a.y[0] > peak                                   # above the peak, not on the line


def test_night_label_keeps_clear_of_the_lines():
    r = at.load_example("level3_day_night")
    for fig in (plots.radicals(r), plots.time_series(r, ["cs_per_s", "h2so4_cm3", "N2O5"], log_y=True)):
        night = [a for a in fig.layout.annotations if a.text == config.NIGHT_LABEL]
        assert night
        for note in night:
            if note.yanchor == "bottom":                   # above the plot
                continue
            lo, hi = fig.layout.yaxis.range
            px = plots._panel_px(fig.layout.height, 3 if "yaxis3" in fig.layout else 1)
            centre = (lo + note.y * (hi - lo))
            for t in fig.data:
                if t.mode != "lines" or t.yaxis not in (None, "y") or t.visible not in (None, True):
                    continue
                x, y = np.asarray(t.x, dtype=float), np.log10(np.asarray(t.y, dtype=float))
                near = np.abs(x - note.x) < 0.3
                gap = np.abs(y[near] - centre) / (hi - lo) * px
                assert np.all(~np.isfinite(gap) | (gap >= 10))


def test_sunlight_labels_name_what_differs():
    assert plots._distinct_parts(["Buffalo, NY, 2026-07-01", "Manaus, Brazil, 2026-07-01"]) == \
        ["Buffalo, NY", "Manaus, Brazil"]
    assert plots._distinct_parts(["Manaus, Brazil, 2026-01-01", "Manaus, Brazil, 2026-07-01"]) == \
        ["2026-01-01", "2026-07-01"]


def test_number_formats_and_no_n_hatching():
    assert [plots._fixed_decimals(v) for v in (9, 5.126, 16.63, 230.4)] == ["9.00", "5.13", "16.6", "230"]
    assert plots._round_down_1(0.39) == 0.3 and plots._round_down_1(0.86) == 0.8
    fig = plots.top_contributors(at.load_example("level2_limonene_soa"))
    n, no_n = fig.data
    assert no_n.name == "no N" and no_n.marker.pattern.shape == "/" and not n.marker.pattern.shape
    assert no_n.marker.color == config.PARTICLE_COLOR       # colours as the notebook text says
