import numpy as np
import pytest

import aaar_tutorial as at
from aaar_tutorial import config


@pytest.fixture(scope="module", params=config.PRESETS)
def example(request):
    return at.load_example(request.param)


def test_example_shapes(example):
    nt, nb = len(example.time_h), len(example.diameter_nm)
    assert example.dndlog10dp_cm3.shape == (nt, nb)
    assert example.dmdlog10dp_ugm3.shape == (nt, nb)
    assert len(example.diameter_edges_nm) == nb + 1
    assert all(v.shape == (nt,) for v in example.totals.values())
    k = len(example.volatility["log10_cstar"])
    assert example.volatility["particle_ugm3"].shape == (nt, k)


def test_example_scenario_round_trips(example):
    s = at.scenario_from(example)
    assert s.name == example.scenario["name"]
    assert s.duration_h == pytest.approx(example.time_h[-1])


def test_unknown_example():
    with pytest.raises(ValueError, match="no example"):
        at.load_example("level9")


def test_series_lookup():
    r = at.load_example("level3_day_night")
    assert at.series(r, "N2O5").shape == (len(r.time_h),)
    assert at.series(r, "cs_per_s").shape == (len(r.time_h),)
    assert "J_NO2" in at.available_series(r)
    with pytest.raises(KeyError, match="available"):
        at.series(r, "NOT_A_SPECIES")


def test_night_intervals():
    assert at.night_intervals(at.load_example("level1_volatility")) == []
    assert at.night_intervals(at.load_example("level2_limonene_soa")) == []   # constant light
    nights = at.night_intervals(at.load_example("level3_day_night"))
    assert len(nights) == 1
    start, end = nights[0]
    assert 8.0 < start < 10.0 and 17.0 < end < 19.0          # Buffalo, 1 July, from noon EDT


def test_summary_matches_instructor_guide():
    s = at.summary(at.load_example("level1_volatility"))      # c* = 1: GMD 54 -> 86 nm
    assert s["gmd_nm_start"] == pytest.approx(54, abs=1)
    assert s["gmd_nm_end"] == pytest.approx(86, abs=1)
    assert s["SOA_ugm3_end"] == pytest.approx(4.93, abs=0.05)
    assert "runtime_s" not in s                                 # nothing ran in this session


def test_derived_series():
    r = at.load_example("level1_volatility")
    np.testing.assert_allclose(at.series(r, "gmd_nm"), at.geometric_mean_diameter_nm(r))
    np.testing.assert_allclose(at.series(r, "org_gas_ugm3"), r.volatility["gas_ugm3"][:, 0])
    assert {"gmd_nm", "org_gas_ugm3"} <= set(at.available_series(r))
    assert at.value_at(r, "SOA_ugm3", r.time_h[-1]) == pytest.approx(at.series(r, "SOA_ugm3")[-1])


def test_night_helpers():
    r = at.load_example("level3_day_night")
    night = at.is_night(r)
    assert night.dtype == bool and 0 < night.sum() < night.size
    assert at.night_mean(r, "N2O5") > 10 * at.series(r, "N2O5")[~night].mean()
    assert not at.is_night(at.load_example("level2_limonene_soa")).any()
    assert np.isnan(at.night_mean(at.load_example("level2_limonene_soa"), "O3"))


def test_unit_helpers():
    r = at.load_example("level3_day_night")                    # 293.15 K, 1 atm
    assert at.ugm3_per_ppb("NO3", r) == pytest.approx(2.58, abs=0.01)
    assert at.ugm3_per_ppb(62.0, r.scenario) == pytest.approx(at.ugm3_per_ppb("NO3", r))
    with pytest.raises(ValueError, match="molar mass"):
        at.ugm3_per_ppb("XYZ", r)
    assert at.cstar_at(1.0, 298.0) == pytest.approx(1.0)
    assert at.cstar_at(1.0, 298.15) == pytest.approx(1.027, abs=0.002)
    assert at.cstar_at(1.0, 278.0) < 0.05


def test_nitrogen_share():
    assert at.nitrogen_share(at.load_example("level2_limonene_soa")) > 0.95
    assert np.isnan(at.nitrogen_share(at.load_example("level1_volatility")))


def test_save_and_load(tmp_path):
    r = at.load_example("level2_limonene_soa")
    back = at.load_result(at.save_result(r, tmp_path / "r.json"))
    np.testing.assert_allclose(back.dndlog10dp_cm3, r.dndlog10dp_cm3)
    assert back.top_contributors == r.top_contributors
