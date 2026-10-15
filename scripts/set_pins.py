"""Change the tutorial ref (Colab badges, install cells) or a model commit everywhere at once.

    python scripts/set_pins.py --tutorial-ref v2026.10
    python scripts/set_pins.py --tomas-jax <40-character commit> [--tuvx-jax <commit>]

Then run pytest: tests/test_config.py and tests/test_notebooks.py check that everything agrees.
A new tomas-jax commit may pin another tuvx-jax commit (its [photolysis] extra); pass that one
with --tuvx-jax so the TUV-x data download matches.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aaar_tutorial import config  # noqa: E402

FILES = ["aaar_tutorial/config.py", "pyproject.toml", "requirements.txt", "README.md",
         "docs/INSTRUCTORS.md", *(f"notebooks/{name}" for name in config.NOTEBOOKS)]
COMMIT = re.compile(r"[0-9a-f]{40}")


def replace_everywhere(pairs: list[tuple[str, str]]) -> None:
    for rel in FILES:
        path = ROOT / rel
        text = path.read_text()
        new = text
        for old, replacement in pairs:
            new = new.replace(old, replacement)
        if new != text:
            path.write_text(new)
            print(f"updated {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tutorial-ref", help="branch or tag of this repository for badges and installs")
    parser.add_argument("--tomas-jax", help="TOMAS-JAX commit (40 hex characters)")
    parser.add_argument("--tuvx-jax", help="tuvx-jax commit (40 hex characters)")
    args = parser.parse_args()
    pairs = []
    if args.tutorial_ref:
        old, new = config.TUTORIAL_REF, args.tutorial_ref
        if "/" in new:      # Colab splits github/<repo>/blob/<ref>/<path> at the first "/"
            parser.error("Colab cannot open a ref that contains '/'; use a tag or a branch without '/'")
        repo = "AAAR-2026-Aerosol-Tutorial"
        pairs += [(f'TUTORIAL_REF = "{old}"', f'TUTORIAL_REF = "{new}"'),
                  (f"{repo}@{old}", f"{repo}@{new}"),
                  (f"{repo}/blob/{old}/", f"{repo}/blob/{new}/"),
                  (f"notebooks from `{old}`", f"notebooks from `{new}`")]
    for flag, old in (("tomas_jax", config.TOMAS_JAX_COMMIT), ("tuvx_jax", config.TUVX_JAX_COMMIT)):
        new = getattr(args, flag)
        if new:
            if not COMMIT.fullmatch(new):
                parser.error(f"--{flag.replace('_', '-')} needs the full 40-character commit")
            pairs += [(old, new), (f"`{old[:7]}`", f"`{new[:7]}`")]
    if not pairs:
        parser.error("nothing to change")
    replace_everywhere(pairs)


if __name__ == "__main__":
    main()
