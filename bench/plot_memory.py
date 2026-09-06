"""
Grouped bar chart of peak memory vs N from the recorded run.

Values are copied from bench/results/rerun_2026-09-06_rtx4060ti.md (the
compare_memory.py table). No measurement happens here. Re-run
compare_memory.py and update the arrays if the numbers change.

    python3 bench/plot_memory.py            # writes docs/profiling/memory.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "profiling" / "memory.png"

N = [1024, 2048, 4096, 8192]
series = [
    ("Custom CUDA (O+L)", [1.0, 2.1, 4.1, 8.2], "#2a78d6"),
    ("PyTorch SDPA Flash", [1.0, 2.1, 4.1, 8.3], "#eb6834"),
    ("HF eager attention", [64.0, 256.0, 1024.0, 4096.0], "#1baf7a"),
]

TEXT, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(N))
w = 0.26
for k, (label, v, c) in enumerate(series):
    xs = x + (k - 1) * w
    ax.bar(xs, v, width=w - 0.03, color=c, label=label, zorder=3)
    for xi, vi in zip(xs, v):
        txt = f"{vi:,.0f}" if vi >= 10 else f"{vi:.1f}"
        ax.text(xi, vi * 1.18, txt, ha="center", va="bottom", fontsize=8, color=TEXT)

ax.set_yscale("log", base=2)
ax.set_ylim(0.5, 12000)
yt = [1, 4, 16, 64, 256, 1024, 4096]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xticks(x)
ax.set_xticklabels([f"{n:,}" for n in N])
ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9, length=0)
ax.set_xlabel("sequence length N", fontsize=9.5, color=MUTED)
ax.set_ylabel("peak MiB above inputs (log scale)", fontsize=9.5, color=MUTED)
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)

fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.13)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white")
print(f"wrote {OUT}")
