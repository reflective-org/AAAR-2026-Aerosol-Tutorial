"""Sites, the mechanism helper and the stale-import check."""
import sys
import types

import pytest

import aaar_tutorial as at
from aaar_tutorial import config, install_check


def test_site_overrides():
    o = at.site_overrides("Buffalo, NY", "2026-07-01")          # 12:00 EDT
    assert o["start_utc"] == "2026-07-01T16:00:00+00:00"
    assert o["photolysis"] == {"latitude_deg": 42.89, "longitude_deg": -78.88}
    assert at.site_overrides("Hyytiälä, Finland", "2026-01-15", 0)["start_utc"].startswith("2026-01-14T22:00")
    with pytest.raises(ValueError, match="config.SITES"):
        at.site_overrides("Nowhere", "2026-07-01")
    at.scenario_from("level3_day_night", **o)                    # valid run() inputs


def test_mechanism_helper(capsys):
    chem = at.mechanism("level2_limonene_soa")
    assert len(chem.mechanism.species) == 726                    # 725 in the file + NA_HET
    assert chem.formula("LIMONENE") == "C10H16"
    chem.show_reactions("LIMONENE")
    out = capsys.readouterr().out
    assert "LIMONENE + O3" in out and "cm³ molec⁻¹ s⁻¹" in out
    assert chem.reactions_of("C822PAN", role="product")
    with pytest.raises(ValueError, match="not a species"):
        chem.reactions_of("NOT_A_SPECIES")


def test_stale_import_is_detected(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", types.SimpleNamespace(__version__="0.0.1"))
    assert any(s.startswith("plotly 0.0.1 ->") for s in install_check.stale_imports())
    with pytest.raises(RuntimeError, match="Restart session"):
        install_check.check_session()


def test_current_session_is_fresh():
    assert install_check.stale_imports() == []
    assert ("numpy", "numpy") in config.UPGRADE_SENSITIVE


@pytest.mark.smoke
def test_sunlight_over_day():
    day = at.sunlight_over_day("Buffalo, NY", "2026-07-01")
    assert day["sza_deg"].min() < 25 and day["sza_deg"].max() > 100
    assert 9e-3 < day["J_NO2"].max() < 1.1e-2 and day["J_NO2"][day["sza_deg"] > 95].max() == 0
