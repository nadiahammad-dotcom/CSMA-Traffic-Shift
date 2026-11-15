import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 200, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777"
ci = json.load(open("capture_isolation.json"))
Ns = [10, 20, 50]; bound = {10: 0.938, 20: 0.936, 50: 0.934}
var = [(0, "Standard BEB", BLUE), (2, "Fixed window, no history", ORANGE), (-2, "Fixed window + random backoff after success", GREEN)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
w = 0.22
for k, (c, lab, col) in enumerate(var):
    J = [[r for r in ci if r["N"] == N and r["G"] == 1.5 and r["cw_mult"] == c][0] for N in Ns]
    x = np.arange(3) + (k - 1) * w
    mk = ["o", "s", "^"][k]
    ax[0].errorbar(x, [r["jain"] for r in J], yerr=[r["jain_sd"] for r in J], fmt=mk, color=col, label=lab, capsize=2, ms=5, elinewidth=0.9)
    ax[1].plot(x, [r["thr"] for r in J], mk, color=col, ms=5)
for a in ax: a.set_xticks(range(3)); a.set_xticklabels([f"N = {n}" for n in Ns])
ax[0].set_ylim(0.75, 1.01); ax[0].set_ylabel("Jain's fairness index (G = 1.5)")
ax[1].set_ylim(0.85, 1.01); ax[1].set_ylabel("Saturation throughput (G = 1.5)")
for i, N in enumerate(Ns):
    ax[1].plot([i - 0.38, i + 0.38], [bound[N]] * 2, color="black", lw=1.0, ls="--")
ax[1].text(2.4, 0.922, "dashed: contention-slot bound", fontsize=6.8, ha="right", va="top")
ax[0].set_title("(a)", loc="left", fontsize=9, fontweight="bold"); ax[1].set_title("(b)", loc="left", fontsize=9, fontweight="bold")
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=3, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig_capture.png"); print("ok")
