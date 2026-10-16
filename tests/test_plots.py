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


def test_level3_night_shading():
    fig = plots.gases(at.load_example("level3_day_night"))
    assert len(fig.layout.shapes) == 1
    assert any("night" in (t.name or "") for t in fig.data)


def test_gases_default_visibility():
    fig = plots.gases(at.load_example("level2_limonene_soa"))
    shown = {t.name for t in fig.data if t.visible is True}
    assert shown == {"Limonene", "O₃", "NO", "NO₂"}


def test_comparisons():
    a = at.load_example("level3_day_night")
    b = at.load_example("level2_limonene_soa")
    fig = plots.totals([a, b], labels=["A", "B"])
    assert {t.name for t in fig.data if t.showlegend} >= {"A", "B"}
    fig = plots.compare_size_distributions([a, b], labels=["A", "B"], time_h=3)
    assert [t.name for t in fig.data] == ["A: initial", "A", "B: initial", "B"]   # different seeds
    assert "t = 3 h" in fig.layout.title.text
    fig.to_json()
    with pytest.raises(ValueError, match="labels"):
        plots.totals([a, b], labels=["only one"])


def test_same_start_gets_one_initial_curve():
    r = at.load_example("level1_volatility")
    fig = plots.compare_size_distributions([r, r], labels=["A", "B"])
    assert [t.name for t in fig.data] == ["initial", "A", "B"]


def test_log_axis_labels_and_gaps():
    r = at.load_example("level3_day_night")
    fig = plots.time_series(r, ["OH", "N_cm3"], log_y=True)
    oh = fig.data[0].y
    assert oh[0] is None or oh[0] != oh[0]                  # t = 0 radical is a gap, not 0
    ticks = fig.layout.yaxis.ticktext
    assert all("<sup>" in t or t.isdigit() for t in ticks)
    assert "<br>" in fig.layout.yaxis.title.text            # unit on its own line


def test_n2o5_is_shown_in_ppt():
    r = at.load_example("level3_day_night")
    fig = plots.time_series(r, ["N2O5"])
    assert fig.data[0].y[-1] == pytest.approx(1e3 * at.series(r, "N2O5")[-1])
    assert "ppt" in fig.layout.yaxis.title.text


def test_new_series_keys_plot():
    r = at.load_example("level1_volatility")
    fig = plots.time_series([r, r], ["SOA_ugm3", "org_gas_ugm3", "gmd_nm"], labels=["a", "b"])
    assert fig.layout.title.text == "SOA, Organic vapor and Geometric mean diameter"


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
    assert not fig.layout.title.subtitle.text
    sub = plots.style(go.Figure(), "my figure", subtitle="details")
    assert sub.layout.title.subtitle.text == "details" and sub.layout.margin.t > fig.layout.margin.t


def test_long_titles_put_details_in_the_subtitle():
    fig = plots.top_contributors(at.load_example("level2_limonene_soa"))
    assert "N-containing" in fig.layout.title.subtitle.text and "N-containing" not in fig.layout.title.text


def test_same_name_runs_get_numbered_labels():
    r = at.load_example("level1_volatility")
    fig = plots.totals([r, r])
    assert {t.name for t in fig.data if t.showlegend} == {"run 1", "run 2"}


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
    assert len(fig.data) == 4                       # one initial and one final curve per run (different grids)
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
    assert "Open Sans" in template.layout.font.family and "Gambarino" in template.layout.title.font.family
    assert list(template.layout.colorway) == list(config.SERIES_COLORS)
    assert template.layout.xaxis.showline and not template.layout.xaxis.mirror
    assert template.layout.xaxis.ticks == "outside" and template.layout.legend.borderwidth == 0
    fig = plots.style(go.Figure(go.Scatter(y=[1, 2])), "mine")
    assert fig.layout.template.layout.font.family == template.layout.font.family
    for fig in (plots.totals(at.load_example("level1_volatility")), plots.partitioning()):
        assert fig.layout.template.layout.title.font.family == config.PLOT_TITLE_FONT_FAMILY
    assert pio.templates.default != config.PLOT_TEMPLATE        # Plotly's global default is untouched


def test_series_differ_in_colour_dash_and_marker():
    n = len(config.SERIES_COLORS)
    assert len(config.SERIES_DASHES) == n and len(config.SERIES_MARKERS) == n + 1   # cycles of different length
    assert len(set(config.SERIES_COLORS)) == len(set(config.SERIES_DASHES)) == n
    assert len(set(config.SERIES_MARKERS)) == n + 1
    assert config.SERIES_COLORS[1] != config.PARTICLE_COLOR      # "contains N" vs "no N" bars
    r = at.load_example("level3_day_night")
    fig = plots.time_series([r, r, r], ["N_cm3"], labels=["a", "b", "c"])
    lines = [t for t in fig.data if t.name in ("a", "b", "c")]
    assert len({t.line.color for t in lines}) == len({t.line.dash for t in lines}) == 3
    assert len({t.marker.symbol for t in lines}) == 3
    assert all(0 < t.marker.maxdisplayed <= 20 for t in lines)                # sparse markers
    assert list(fig.layout.xaxis.range) == [r.time_h[0], r.time_h[-1]]      # no padding for markers


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


def _triple(trace):
    return (trace.line.color, trace.line.dash, trace.marker.symbol)


@pytest.mark.parametrize("make", [plots.gases, plots.radicals])
def test_every_species_line_has_its_own_style(example, make):
    fig = make(example)
    lines = [t for t in fig.data if "night" not in (t.name or "")]
    if not example.gases_ppb:
        return                                              # Level 1: message figure
    triples = [_triple(t) for t in lines]
    assert len(set(triples)) == len(triples), triples


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
    comp = {t.name: t.fillcolor for t in plots.composition(r).data if t.fillcolor}
    assert comp["SOA"] == colors["SOA_ugm3"] and comp["Sulfate"] == colors["SO4_ugm3"]
    assert all(t.line.dash == "solid" for t in fig.data if t.mode == "lines")


def test_template_cycles_dashes_and_internal_lines_stay_solid():
    template = pio.templates[config.PLOT_TEMPLATE]
    assert [t.line.dash for t in template.data.scatter] == list(config.SERIES_DASHES)
    r = at.load_example("level3_day_night")
    fig = plots.size_distribution(r)
    assert all(t.line.dash in ("solid", "dot") for t in fig.data)
    for fig in (plots.composition(r), plots.photolysis(r)):
        assert all(t.line.dash is not None for t in fig.data)


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
    assert "Iowan Old Style" in config.PLOT_TITLE_FONT_FAMILY.split("Georgia")[0]


def test_composition_night_bands_show_over_the_stack():
    r = at.load_example("level3_day_night")
    fig = plots.composition(r, fraction=True)
    assert fig.layout.shapes and all(sh.layer == "above" for sh in fig.layout.shapes)
    assert list(fig.layout.yaxis.range) == [0, 100]


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
    plots.style(fig, "panels")
    assert fig.layout.legend.yref == "container" and fig.layout.legend.yanchor == "top"
    assert fig.layout.margin.t > 90
