"""Sensitivity of the forecasting result to the congestion definition (N = 20, base traffic runs).
Backlog threshold Q* in {N, 2N, 4N} frames; horizon h = 5; logistic regression versus the utilisation rule."""
import json, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings("ignore")
L, N, K, H = 24, 20, 8, 5
D = json.load(open("series.json"))
TRAIN, VAL, TEST = list(range(0, 24)), list(range(24, 30)), list(range(30, 40))

def feats(r):
    thr, coll = np.array(D["thr"][r]), np.array(D["coll"][r]); q = np.array(D["qlen"][r], float)
    sr = np.maximum(thr / L, 1e-9)
    return np.stack([thr, coll * 10, np.minimum(coll / sr, 5.0)], 1), q

def build(idx, qstar, onset):
    X, y, u = [], [], []
    for r in idx:
        b, q = feats(r)
        for i in range(K - 1, len(q) - H):
            if onset and q[i] >= qstar: continue
            X.append(b[i - K + 1:i + 1].ravel()); y.append(int(q[i + H] >= qstar)); u.append(b[i, 0])
    return np.array(X), np.array(y), np.array(u)

def best_thr(y, s):
    cs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    return cs[int(np.argmax([f1_score(y, s >= c, zero_division=0) for c in cs]))]

out = []
for mult in (1, 2, 4):
    qs = mult * N
    for onset in (False, True):
        Xtr, ytr, _ = build(TRAIN, qs, onset); Xva, yva, _ = build(VAL, qs, onset); Xte, yte, ute = build(TEST, qs, onset)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
        sva, ste = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        t = best_thr(yva, sva)
        row = {"qstar": qs, "mult": mult, "task": "onset" if onset else "state", "n_pos": int(yte.sum()), "n": int(len(yte)),
               "logit_auc": float(roc_auc_score(yte, ste)), "logit_f1": float(f1_score(yte, ste >= t)),
               "util_auc": float(roc_auc_score(yte, ute)), "util_f1": float(f1_score(yte, ute >= 0.85))}
        out.append(row); print(row, flush=True)
json.dump(out, open("threshold_sensitivity.json", "w"), indent=1)
