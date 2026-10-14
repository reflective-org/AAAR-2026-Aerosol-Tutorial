"""Runs the real model once (deselect with: pytest -m "not smoke")."""
import pytest

import aaar_tutorial as at
from aaar_tutorial import plots


@pytest.mark.smoke
def test_level1_short_run():
    result = at.run("level1_volatility", duration_h=1.0, progress=False, name="smoke")
    assert len(result.time_h) == 7                       # output every 10 min
    assert result.scenario["name"] == "smoke"
    assert 0 < result.totals["SOA_ugm3"][-1] < 1.0       # 0.5 ug/m3/h for 1 h
    assert result.diagnostics["warnings"] == []
    plots.size_distribution(result).to_json()
