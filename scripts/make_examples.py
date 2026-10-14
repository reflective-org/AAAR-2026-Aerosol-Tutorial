"""Regenerate the stored preset results in aaar_tutorial/examples/ with the installed model.

    python scripts/make_examples.py            # all presets, about 1 minute

Run it after changing the TOMAS-JAX pin. Floats are rounded to 6 significant digits, like
the web app's fallback results, to keep the files small.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import aaar_tutorial as at  # noqa: E402
from aaar_tutorial import config  # noqa: E402

SIGNIFICANT_DIGITS = 6


def rounded(obj):
    if isinstance(obj, float) and math.isfinite(obj) and obj != 0.0:
        return float(f"{obj:.{SIGNIFICANT_DIGITS}g}")
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    return obj


def main() -> None:
    at.setup()
    for preset in config.PRESETS:
        result = at.run(preset, progress=False)
        path = ROOT / "aaar_tutorial" / "examples" / f"{preset}.json"
        path.write_text(json.dumps(rounded(result.to_dict()), separators=(",", ":")))
        print(f"{path.relative_to(ROOT)}: {result.diagnostics['runtime_s']:.1f} s, "
              f"code version {result.diagnostics['code_version']}")


if __name__ == "__main__":
    main()
