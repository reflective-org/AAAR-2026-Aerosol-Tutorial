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


def test_seed_shorthand_changes_the_first_mode():
    s = at.scenario_from("level1_volatility", seed={"number_cm3": 5000, "organic_mass_fraction": 0.0})
    assert s.seed.modes[0].number_cm3 == 5000 and s.seed.modes[0].gmd_nm == 50.0
    assert s.seed.organic_mass_fraction == 0.0
    s = at.scenario_from("level1_volatility", seed={"organic_mass_fraction": 0.6})
    assert s.seed.modes[0].number_cm3 == 2000.0
    with pytest.raises(ValueError, match="not both"):
        at.scenario_from("level1_volatility", seed={"gmd_nm": 80, "modes": []})


def test_cstar_table_is_written_once(tmp_path, monkeypatch):
    from aaar_tutorial import config
    monkeypatch.setattr(config, "CSTAR_TABLE_DIR", tmp_path)
    name = at.write_cstar_table({"B": 2.0, "A": 1e5})
    assert name == at.write_cstar_table({"A": 1e5, "B": 2.0})
    assert (tmp_path / name).read_text() == "species,cstar_ugm3\nA,100000\nB,2\n"
    with pytest.raises(ValueError, match="gas chemistry"):
        at.run("level1_volatility", cstar_ugm3={"A": 1.0}, progress=False)


@pytest.mark.smoke
def test_progress_callable_gets_fractions():
    seen = []
    at.run("level1_volatility", duration_h=0.5, progress=seen.append)
    assert seen[-1] == pytest.approx(1.0) and len(seen) == 6
