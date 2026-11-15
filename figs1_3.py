import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 200,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
BLUE, ORANGE, GREEN, VERM, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777", "#CC79A7"
sw = json.load(open("sweep.json")); fc = json.load(open("forecast.json"))

# ---------- Figure 1: load sweep ----------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
cols = {10: BLUE, 20: ORANGE, 50: GREEN}
for N, c in cols.items():
    rows = [x for x in sw if x["N"] == N]
    G = np.array([x["G"] for x in rows])
    for k, key in enumerate(["throughput", "p_coll_per_event", "mean_delay_slots"]):
        y = np.array([x[key] for x in rows]); sd = np.array([x[key + "_sd"] for x in rows])
        ax[k].plot(G, y, color=c, lw=1.4, label=f"N = {N}", zorder=3)
        ax[k].errorbar(G, y, yerr=sd, fmt="none", ecolor=c, elinewidth=0.7, capsize=1.5, zorder=3)
        ax[k].scatter(G[G < 1], y[G < 1], s=11, color=c, zorder=4)
        ax[k].scatter(G[G >= 1], y[G >= 1], s=11, facecolors="white", edgecolors=c, linewidths=0.9, zorder=4)
ax[0].plot([0, 1.5], [0, 1.5], color=GREY, lw=0.8, ls="--"); ax[0].set_ylim(0, 1.05)
ax[0].text(0.12, 0.45, "ideal\n(throughput = load)", color=GREY, fontsize=7, rotation=0)
ax[0].set_ylabel("Throughput (fraction of capacity)")
ax[1].set_ylabel("Collisions per contention event"); ax[1].set_ylim(0, 0.45)
ax[2].set_yscale("log"); ax[2].set_ylabel("Mean access delay (slots)")
for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
    a.set_xlabel("Offered load G"); a.set_title(t, loc="left", fontsize=9, fontweight="bold"); a.set_xlim(0, 1.55)
ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)
ax[2].text(1.02, 2, "open markers: load ≥ 1,\ndelay bounded by run length", fontsize=6.5, color=GREY)
fig.tight_layout(); fig.savefig("fig1_sweep.png"); plt.close(fig)

# ---------- Figure 2: one test run ----------
D = json.load(open("series.json")); L = 24; K = 8; QSTAR = 40; h = 5
def feats(run):
    thr, coll = np.array(D["thr"][run]), np.array(D["coll"][run])
    sr = np.maximum(thr / L, 1e-9)
    return np.stack([thr, coll * 10, np.minimum(coll / sr, 5.0)], 1), np.array(D["qlen"][run], float)
X, y = [], []
for r in range(0, 24):
    b, q = feats(r)
    for i in range(K - 1, len(q) - h):
        X.append(b[i - K + 1:i + 1].ravel()); y.append(int(q[i + h] >= QSTAR))
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.array(X), np.array(y))
run = 37
b, q = feats(run); thr = b[:, 0]; G = np.array(D["G"][run])
p = np.full(len(q), np.nan)
for i in range(K - 1, len(q) - h):
    p[i + h] = m.predict_proba(b[i - K + 1:i + 1].ravel().reshape(1, -1))[0, 1]   # forecast issued h windows earlier
sl = slice(40, 240); x = np.arange(len(q))[sl]
fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.1, 0.9]})
ax[0].step(x, G[sl], where="post", color=GREY, lw=1.2, label="Offered load (hidden from the hub)")
ax[0].plot(x, thr[sl], color=BLUE, lw=1.0, label="Measured throughput")
ax[0].set_ylabel("Fraction of capacity"); ax[0].legend(frameon=False, ncol=2, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax[1].plot(x, np.maximum(q[sl], 1), color=VERM, lw=1.2); ax[1].set_yscale("log")
ax[1].axhline(QSTAR, color="black", lw=0.8, ls="--"); ax[1].text(x[-1], QSTAR * 1.15, "congestion threshold (40 frames)", ha="right", fontsize=7)
ax[1].fill_between(x, 1, 8000, where=q[sl] >= QSTAR, color=VERM, alpha=0.10, lw=0); ax[1].set_ylim(1, 8000)
ax[1].set_ylabel("Backlog (frames)")
ax[2].plot(x, p[sl], color=PURPLE, lw=1.1); ax[2].axhline(0.5, color="black", lw=0.8, ls="--")
ax[2].fill_between(x, 0, 1, where=q[sl] >= QSTAR, color=VERM, alpha=0.10, lw=0)
ax[2].set_ylim(0, 1.02); ax[2].set_ylabel(f"Forecast P(congested)\n(issued {h} windows earlier)"); ax[2].set_xlabel("Window index (500 slots each)")
for a, t in zip(ax, ["(a)", "(b)", "(c)"]): a.set_title(t, loc="left", fontsize=9, fontweight="bold", pad=2)
fig.tight_layout(); fig.savefig("fig2_timeseries.png"); plt.close(fig)

print("ok")
