"""Figure 5: (a) transfer across traffic processes, (b) false alarms at 80% window recall (reads strict_eval.json)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 200, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})
D = json.load(open("strict_eval.json"))
def get(sc, test, task, h): return [r for r in D if r["scenario"] == sc and r["test"] == test and r["task"] == task and r["h"] == h][0]
conds = [("A", "base2", "train base\ntest base"), ("B", "burst", "train base\ntest burst"), ("C", "base", "train burst\ntest base"), ("C", "burst", "train burst\ntest burst")]
models = [("Utilisation rule", "#E69F00", "Utilisation rule"), ("Logistic (8 windows)", "#0072B2", "Logistic regression"),
          ("GradBoost", "#009E73", "Gradient boosting"), ("LSTM", "#CC79A7", "LSTM"), ("Persistence (needs backlog)", "#777777", "Persistence of true state (not observable)")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]; w = 0.16
for j, (k, c, lab) in enumerate(models):
    vals = [get(sc, t, "state", 5)[k]["auc"] for sc, t, _ in conds]
    a.bar(np.arange(4) + (j - 2) * w, vals, w, color=c, label=lab)
a.set_xticks(range(4)); a.set_xticklabels([x[2] for x in conds], fontsize=7.5); a.set_ylim(0.7, 1.0); a.set_ylabel("ROC AUC, congested state, h = 5")
a.set_title("(a) Transfer across traffic processes", loc="left", fontsize=8.5, fontweight="bold")
b = ax[1]; w = 0.2
labs = [("A", "base2", 5, "h=5"), ("A", "base2", 10, "h=10")]
mm = [m for m in models[:4]]
for j, (k, c, lab) in enumerate(mm):
    vals = [get(sc, t, "onset", h)[k]["events"]["false_alarm_per_100"] for sc, t, h, _ in labs]
    b.bar(np.arange(2) + (j - 1.5) * w, vals, w, color=c)
b.set_xticks(range(2)); b.set_xticklabels(["h = 5", "h = 10"]); b.set_ylabel("False alarms per 100 windows")
b.set_title("(b) Onset alarms at ~80% recall", loc="left", fontsize=8.5, fontweight="bold")
fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig("fig5_transfer.png"); print("ok")
