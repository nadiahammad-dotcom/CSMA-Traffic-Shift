"""Figure 4: ROC AUC with 95% bootstrap intervals. (a) congested state by horizon, all predictors (forecast_seq_base.json, test runs 30-39);
(b) onset within h windows, 40 further runs (strict_eval.json, scenario A, test set base2)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "figure.dpi": 200, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})
fc = [r for r in json.load(open("forecast_seq_base.json")) if r["task"] == "state"]
se = [r for r in json.load(open("strict_eval.json")) if r["scenario"] == "A" and r["test"] == "base2" and r["task"] == "onset"]
sty = {"Oracle load rule": ("#777777", "o", "Oracle load rule (not observable)"), "Utilisation rule": ("#E69F00", "s", "Utilisation rule"),
       "Logistic": ("#0072B2", "^", "Logistic regression"), "Logistic (8 windows)": ("#0072B2", "^", "Logistic regression"),
       "Logistic (last window)": ("#56B4E9", "v", "Logistic, last window only"), "GradBoost": ("#009E73", "D", "Gradient boosting"),
       "MLP": ("#D55E00", "<", "Multilayer perceptron"), "LSTM": ("#CC79A7", "P", "LSTM"), "GRU": ("#999933", "X", "GRU")}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
def draw(a, rows, hs, names, ttl):
    off = np.linspace(-0.34, 0.34, len(names))
    for nm, o in zip(names, off):
        c, mk, lab = sty[nm]
        auc = np.array([r[nm]["auc"] for r in rows]); lo = np.array([r[nm]["auc_lo"] for r in rows]); hi = np.array([r[nm]["auc_hi"] for r in rows])
        a.errorbar(np.arange(len(hs)) + o, auc, yerr=[auc - lo, hi - auc], fmt=mk, color=c, ms=3.8, elinewidth=0.8, capsize=1.2, label=lab)
    a.set_xticks(range(len(hs))); a.set_xticklabels(hs); a.set_xlabel("Forecast horizon h (windows)")
    a.set_title(ttl, loc="left", fontsize=8.5, fontweight="bold")
draw(ax[0], fc, [1, 3, 5, 10], ["Oracle load rule", "Utilisation rule", "Logistic", "GradBoost", "MLP", "LSTM", "GRU"], "(a) Congested state at t + h")
draw(ax[1], se, [5, 10], ["Utilisation rule", "Logistic (last window)", "Logistic (8 windows)", "GradBoost", "LSTM"], "(b) Onset within h windows")
ax[0].set_ylim(0.78, 1.005); ax[1].set_ylim(0.85, 1.0); ax[0].set_ylabel("ROC AUC")
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
d = dict(zip(l1, h1)); d.update(dict(zip(l2, h2)))
fig.legend(d.values(), d.keys(), frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handletextpad=0.2)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig("fig3_auc.png"); print("ok")
