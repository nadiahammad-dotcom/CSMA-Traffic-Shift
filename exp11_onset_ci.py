"""Run-level bootstrap intervals for the onset operating points of Table 9 (scenario A, test set base2;
logistic regression, gradient boosting, utilisation rule). Reuses the definitions in exp9_strict.py (lines between the
constants and fit_lstm) so that data, features, training runs and the 80%-recall threshold rule are identical; LSTM omitted for speed."""
import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
src = open("exp9_strict.py").read()
ns = {"json": json, "np": np, "roc_auc_score": roc_auc_score}
exec(src[src.index("N, L, K = 20, 24, 8"):src.index("def fit_lstm")], ns)
build, thr_for_recall = ns["build"], ns["thr_for_recall"]
TR, VA = range(0, 24), range(24, 30)
out = []
rng = np.random.default_rng(0)
for h in (5, 10):
    Xtr, ytr, _, _, _ = build("base", TR, h, "onset"); Xva, yva, _, _, _ = build("base", VA, h, "onset")
    Xte, yte, rte, _, _ = build("base2", range(40), h, "onset")
    flat = lambda A: A.reshape(len(A), -1)
    sc = {}
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(flat(Xtr), ytr); sc["Logistic regression"] = (m.predict_proba(flat(Xva))[:, 1], m.predict_proba(flat(Xte))[:, 1])
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(flat(Xtr), ytr); sc["Gradient boosting"] = (m.predict_proba(flat(Xva))[:, 1], m.predict_proba(flat(Xte))[:, 1])
    sc["Utilisation rule"] = (Xva[:, -1, 0], Xte[:, -1, 0])
    ids = np.unique(rte); idx = {p: np.where(rte == p)[0] for p in ids}
    for nm, (sva, ste) in sc.items():
        t = thr_for_recall(yva, sva, 0.8); pred = ste >= t
        def met(m_):
            y, p = yte[m_], pred[m_]; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = ((~p) & (y == 1)).sum()
            return tp / max(1, tp + fn), tp / max(1, tp + fp), 100 * fp / len(y)
        point = met(np.arange(len(yte))); bs = []
        for _ in range(500):
            mm = np.concatenate([idx[p] for p in rng.choice(ids, len(ids))]); bs.append(met(mm))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        out.append({"h": h, "model": nm, "recall": point[0], "precision": point[1], "fa_per_100": point[2],
                    "recall_ci": [lo[0], hi[0]], "precision_ci": [lo[1], hi[1]], "fa_ci": [lo[2], hi[2]]})
        print(h, nm, "rec %.2f [%.2f,%.2f] prec %.2f [%.2f,%.2f] FA %.1f [%.1f,%.1f]" % (point[0], lo[0], hi[0], point[1], lo[1], hi[1], point[2], lo[2], hi[2]), flush=True)
json.dump(out, open("onset_ci.json", "w"), indent=1)
