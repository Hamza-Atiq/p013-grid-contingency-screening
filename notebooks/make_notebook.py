"""Write notebooks/p013_live.ipynb, the notebook behind the 'Run it live' link.

Generated rather than hand-edited so the cells stay reviewable as plain
Python. Run from the repository root:

    python notebooks/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent / "p013_live.ipynb"

INTRO = """# P-013: if one power line fails, does anything overload?

An AI ranks all 80 possible single line failures on the IEEE 57-bus grid, and the
real Newton-Raphson solver (pandapower) checks only the risky ones.
**The AI sorts, the solver decides.**

**To run it:** click **Runtime > Run all**. The first cell downloads the code and
installs pandapower, numba and Gradio (about a minute). The second screens one
day and checks all 80 cases with the solver. The third opens the interactive app
right here in the notebook.

Recorded results for 175 test days: https://hamza-atiq.github.io/p013-grid-contingency-screening/
Code and full write-up: https://github.com/Hamza-Atiq/p013-grid-contingency-screening
"""

SETUP = """# 1. Get the code and install what it needs (about a minute on Colab).
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/Hamza-Atiq/p013-grid-contingency-screening"
if not Path("space/app.py").exists():              # on Colab: fetch the repository
    if not Path("p013-grid-contingency-screening").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
    os.chdir("p013-grid-contingency-screening")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pandapower==3.5.4", "numba", "networkx", "gradio==6.29.1"], check=True)
print("Ready in", os.getcwd())"""

ONE_DAY = """# 2. Screen one day: total load 1,000 MW, seed 1. Then solve all 80 to check.
import importlib.util
from IPython.display import Markdown, display

spec = importlib.util.spec_from_file_location("p013_app", "space/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
app.warm_up()                                      # compiles the solver once

summary, table = app.check_all(1000, 1)
display(Markdown(summary))
import pandas as pd
pd.DataFrame(table, columns=app.HEADERS).head(20)"""

NOTE = """Scores here come from the same trained model and the same policy file as the
recorded results. Colab's NumPy build can differ from the laptop the results
were recorded on, so a score may differ in its last digits. That can only change
which lines are sent if a score sits within a hair of the cutoff.

## 3. Try your own days

Move the load slider, change the seed (each seed is a different day), and press
**Run screening**. **Check all 80 with the solver** shows whether the AI missed anything."""

LAUNCH = """# 3. The interactive app, inline.
if not os.environ.get("P013_NO_LAUNCH"):
    app.build().queue(default_concurrency_limit=1).launch()"""


def main() -> None:
    nb = new_notebook(cells=[
        new_markdown_cell(INTRO),
        new_code_cell(SETUP),
        new_code_cell(ONE_DAY),
        new_markdown_cell(NOTE),
        new_code_cell(LAUNCH),
    ])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": [], "name": "p013_live.ipynb"}
    nbformat.write(nb, OUT)
    print(f"written {OUT.name}")


if __name__ == "__main__":
    main()
