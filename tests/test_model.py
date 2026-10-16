import math

import pytest

import aaar_tutorial as at


def test_overrides_merge_gases_and_nested_sections():
    s = at.scenario_from("level2_limonene_soa", gases_ppb={"LIMONENE": 20.0},
                         environment={"rh": 0.8}, name="more limonene")
    assert s.gases_ppb["LIMONENE"] == 20.0
    assert s.gases_ppb["O3"] == 40.0                     # untouched species keep preset values
    assert s.environment.rh == 0.8 and s.environment.temp_k == 298.15
    assert s.name == "more limonene"


def test_unknown_preset():
    with pytest.raises(ValueError, match="choose one of"):
        at.scenario_from("level4")


def test_describe_prints_sections(capsys):
    at.describe("level3_day_night")
    out = capsys.readouterr().out
    assert "gamma_n2o5 = 0.02" in out and "environment: temp_k=293.15" in out


def test_timeout_stops_run():
    with pytest.raises(TimeoutError, match="timeout_s"):
        at.run("level1_volatility", duration_h=1.0, progress=False, timeout_s=0.0)


def test_bad_duration_is_reported():
    with pytest.raises(ValueError, match="multiple of the coupling interval"):
        at.run("level1_volatility", duration_h=0.01, progress=False)


def test_unknown_input_suggests_the_right_spelling():
    with pytest.raises(ValueError, match=r"surrogate=\{'cstar_ugm3'"):
        at.scenario_from("level1_volatility", cstar=10)
    with pytest.raises(ValueError, match=r"environment=\{'rh'"):
        at.scenario_from("level1_volatility", rh=0.8)
    with pytest.raises(ValueError, match=r"environment=\{'temp_k'"):
        at.scenario_from("level1_volatility", environment={"temp": 280})


def test_model_input_checks_reach_the_student_unchanged():
    """The model checks values (TOMAS-JAX box/scenario.py); the helpers pass its message on as it is."""
    from tomas_jax.box.scenario import Environment, Processes

    with pytest.raises(ValueError) as direct:
        Environment(rh=80)
    for call in (lambda: at.scenario_from("level1_volatility", environment={"rh": 80}),
                 lambda: at.run("level1_volatility", progress=False, environment={"rh": 80})):
        with pytest.raises(ValueError) as via_helpers:
            call()
        assert str(via_helpers.value) == str(direct.value)
    assert "did you mean 0.8?" in str(direct.value)

    with pytest.raises(ValueError) as direct:
        Processes(coagulation="off")
    with pytest.raises(ValueError) as via_helpers:
        at.run("level1_volatility", progress=False, processes={"coagulation": "off"})
    assert str(via_helpers.value) == str(direct.value)
    assert str(direct.value) == "processes.coagulation must be True or False, got 'off'"


def test_seed_shorthand_changes_the_first_mode():
    s = at.scenario_from("level1_volatility", seed={"number_cm3": 5000, "organic_mass_fraction": 0.0})
    assert s.seed.modes[0].number_cm3 == 5000 and s.seed.modes[0].gmd_nm == 50.0
    assert s.seed.organic_mass_fraction == 0.0
    s = at.scenario_from("level1_volatility", seed={"organic_mass_fraction": 0.6})
    assert s.seed.modes[0].number_cm3 == 2000.0
    with pytest.raises(ValueError, match="not both"):
        at.scenario_from("level1_volatility", seed={"gmd_nm": 80, "modes": []})


def _default_cstar_table():
    from tomas_jax.gaschem.config import DEFAULT_CSTAR_TABLE, MECHANISM_DIR
    from tomas_jax.gaschem.volatility import load_cstar_table
    return load_cstar_table(MECHANISM_DIR / DEFAULT_CSTAR_TABLE)


def test_cstar_table_is_written_once(tmp_path, monkeypatch):
    from aaar_tutorial import config
    monkeypatch.setattr(config, "CSTAR_TABLE_DIR", tmp_path)
    name = at.write_cstar_table({"LIMALOOH": 2.0, "LIMAL": 1e5})
    assert name == at.write_cstar_table({"LIMAL": 1e5, "LIMALOOH": 2.0})
    assert [p.name for p in tmp_path.iterdir()] == [name]
    with pytest.raises(ValueError, match="gas chemistry"):
        at.run("level1_volatility", cstar_ugm3={"LIMAL": 1.0}, progress=False)


def test_cstar_table_keeps_the_default_values(tmp_path, monkeypatch):
    """A 2-species user table is merged into the default table; every other row is kept."""
    from tomas_jax.gaschem.volatility import load_cstar_table
    from aaar_tutorial import config
    monkeypatch.setattr(config, "CSTAR_TABLE_DIR", tmp_path)
    default = _default_cstar_table()
    written = load_cstar_table(tmp_path / at.write_cstar_table({"LIMALOOH": 2.0, "NEWSPECIES": 30.0}))
    assert set(written) == set(default) | {"NEWSPECIES"}
    assert written["LIMALOOH"] == pytest.approx(math.log10(2.0), abs=1e-12)
    assert written["NEWSPECIES"] == pytest.approx(math.log10(30.0), abs=1e-12)
    for species, log10_cstar in default.items():
        if species != "LIMALOOH":
            assert written[species] == pytest.approx(log10_cstar, abs=1e-12), species


@pytest.mark.smoke
def test_run_uses_the_merged_cstar_table(tmp_path, monkeypatch):
    """run(cstar_ugm3=...) changes the listed species only, with no partial-table warning."""
    from aaar_tutorial import config
    monkeypatch.setattr(config, "CSTAR_TABLE_DIR", tmp_path)
    default = _default_cstar_table()
    r = at.run("level2_limonene_soa", duration_h=1.0, progress=False,
               cstar_ugm3={"LIMALOOH": 1.0, "C826OOH": 0.1})
    assert not [w for w in r.diagnostics["warnings"] if "c* table" in w]
    source = r.diagnostics["cstar_source"]
    assert source.startswith(f"table {r.scenario['cstar_table']} ")
    assert "0 of them condensable" in source            # no species falls back to Li et al.
    top = {c["species"]: c["log10_cstar"] for c in r.top_contributors}
    assert top["LIMALOOH"] == pytest.approx(0.0, abs=1e-12)
    assert top["C826OOH"] == pytest.approx(-1.0, abs=1e-12)
    others = [s for s in top if s not in ("LIMALOOH", "C826OOH")]
    assert others
    for species in others:
        assert top[species] == pytest.approx(default[species], abs=1e-12), species


@pytest.mark.smoke
def test_progress_callable_gets_fractions():
    seen = []
    at.run("level1_volatility", duration_h=0.5, progress=seen.append)
    assert seen[-1] == pytest.approx(1.0) and len(seen) == 6


def test_presets_run_on_the_notebook_bin_count():
    from tomas_jax.box.scenario import load_preset
    s = at.scenario_from("level3_day_night")
    assert s.numerics.nbins == at.config.NBINS
    assert s.numerics.coupling_dt_s == load_preset("level3_day_night").numerics.coupling_dt_s
    assert at.scenario_from("level3_day_night", numerics={"nbins": 40}).numerics.nbins == 40
    assert at.scenario_from(load_preset("level1_volatility")).numerics.nbins == 40    # a Scenario keeps its own


def test_use_bins_sets_the_default_for_preset_runs():
    default = at.config.NBINS
    try:
        at.use_bins(160)
        assert at.scenario_from("level1_volatility").numerics.nbins == 160
        assert at.scenario_from("level1_volatility", numerics={"nbins": 80}).numerics.nbins == 80
        with pytest.raises(ValueError, match="one of"):
            at.use_bins(100)
    finally:
        at.use_bins(default)
    assert at.config.NBINS == 40
