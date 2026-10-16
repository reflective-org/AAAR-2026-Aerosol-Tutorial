import numpy as np
import plotly.graph_objects as go
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


def _scale_menu(fig):
    menus = [m for m in fig.layout.updatemenus if m.buttons and m.buttons[-1].label.startswith("Log")]
    assert len(menus) == 1
    return menus[0]


def test_banana_white_floor_and_colour_buttons():
    r = at.load_example("level1_volatility")
    fig = plots.banana(r)
    assert fig.data[0].colorscale[0][1].lower() == "#ffffff"
    assert not np.isnan(np.asarray(fig.data[0].z, dtype=float)).any()      # empty cells drawn white
    menu = _scale_menu(fig)
    assert [b.label for b in menu.buttons] == ["Linear colour", "Log colour"] and menu.active == 1
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
    assert [b.label for b in menu.buttons] == ["Linear", "Log"] and menu.active == int(log_first)
    layouts = [b.args[0] if b.method == "relayout" else b.args[1] for b in menu.buttons]
    assert sum(k.endswith(".type") for k in layouts[1]) == n_axes
    assert all(v == "log" for k, v in layouts[1].items() if k.endswith(".type"))
    assert all(v == "linear" for k, v in layouts[0].items() if k.endswith(".type"))


def test_log_button_turns_zeros_into_gaps():
    r = at.load_example("level3_day_night")
    menu = _scale_menu(plots.radicals(r))
    lin_y, log_y = (b.args[0]["y"] for b in menu.buttons)
    assert lin_y[0][0] == 0.0 and np.isnan(log_y[0][0])                    # OH at t = 0
