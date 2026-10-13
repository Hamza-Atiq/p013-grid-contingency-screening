"""Assemble the Hugging Face Space in space/_build/.

Copies only what the Space runs: the app, the p013 package, the case file,
the trained weights and the frozen live policy. The layout keeps
``cases/`` two levels above ``src/p013/`` so ``p013.CASE_PATH`` resolves.

    .venv/Scripts/python.exe space/build_space.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
OUT = SPACE / "_build"

FILES = {
    SPACE / "app.py": "app.py",
    SPACE / "README.md": "README.md",
    SPACE / "requirements.txt": "requirements.txt",
    ROOT / "LICENSE": "LICENSE",
    ROOT / "model" / "weights.npz": "model/weights.npz",
    ROOT / "model" / "live_policy.npz": "model/live_policy.npz",
    ROOT / "cases" / "pglib_opf_case57_ieee.m": "cases/pglib_opf_case57_ieee.m",
}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for src, dst in FILES.items():
        target = OUT / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    shutil.copytree(ROOT / "src" / "p013", OUT / "src" / "p013",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT.relative_to(ROOT)}: {len(files)} files")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
