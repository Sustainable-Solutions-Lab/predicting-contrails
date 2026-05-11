"""Mirror the latest repo-generated figures into the Dropbox Plots subfolder.

The lab Dropbox has one subfolder per figure (`Plots/Figure 1/`, ..., 4).
This script copies each repo-generated figure (PNG + PDF) into the
corresponding subfolder so the outputs sit alongside Silas's existing
files. Doesn't overwrite anything except its own prior copies, since the
filenames are distinct (`fig1_map.png`, `fig3c.png`, ...).

Run after regenerating any figure:

    /Users/stevedavis/anaconda3/bin/python figures/sync_to_dropbox.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

DROPBOX_PLOTS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/WS Corp contrails (w Silas)/Plots"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP_OUT = REPO / "experiments" / "outputs"

# (figure_number, source_paths)
MAPPING = [
    (1, [HERE / "outputs" / "fig1_map.png",
         HERE / "outputs" / "fig1_map.pdf",
         HERE / "outputs" / "fig1_map.eps"]),
    (3, [EXP_OUT / "fig3c.png",
         EXP_OUT / "fig3c.pdf",
         EXP_OUT / "fig3c.eps"]),
    (4, [HERE / "customer_outputs" / "fig4_combined.png",
         HERE / "customer_outputs" / "fig4_combined.pdf",
         HERE / "customer_outputs" / "fig4_combined.eps",
         HERE / "customer_outputs" / "fig4_summary.csv"]),
    (5, [HERE / "customer_outputs" / "fig5_demand_shift.png",
         HERE / "customer_outputs" / "fig5_demand_shift.pdf",
         HERE / "customer_outputs" / "fig5_demand_shift.eps",
         HERE / "customer_outputs" / "fig5_per_flight.csv"]),
]


def main() -> None:
    if not DROPBOX_PLOTS.exists():
        raise SystemExit(f"Dropbox Plots folder not found: {DROPBOX_PLOTS}")

    for fig_num, srcs in MAPPING:
        target = DROPBOX_PLOTS / f"Figure {fig_num}"
        target.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            if not src.exists():
                print(f"  skip  {src.relative_to(REPO)} (not found)")
                continue
            dst = target / src.name
            shutil.copy2(src, dst)
            print(f"  copy  {src.relative_to(REPO)}  →  Plots/Figure {fig_num}/{src.name}")


if __name__ == "__main__":
    main()
