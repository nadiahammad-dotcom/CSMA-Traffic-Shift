"""Operating-point sensitivity for onset alarms, paired AUC difference against persistence, and Brier scores.
Scenario A of exp9_strict.py (train runs 0-23 of the base process, validation 24-29, test = 40 further runs 'base2'). LSTM omitted.
Output: onset_ops.json, state_extra.json"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
src = open("exp9_strict.py").read()
ns = {"json": json, "np": np, "roc_auc_score": roc_auc_score}
exec(src[src.index("N, L, K = 20, 24, 8"):src.index("def fit_lstm")], ns)
build, thr_for_recall, event_metrics = ns["build"], ns["thr_for_recall"], ns["event_metrics"]
TR, VA = range(0, 24), range(24, 30)
rng = np.random.default_rng(1)
flat = lambda A: A.reshape(len(A), -1)
def models(Xtr, ytr):
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(flat(Xtr), ytr)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(flat(Xtr), ytr)
    return {"Logistic regression": lambda A: lg.predict_proba(flat(A))[:, 1], "Gradient boosting": lambda A: gb.predict_proba(flat(A))[:, 1], "Utilisation rule": lambda A: A[:, -1, 0]}
ops = []
for h in (5, 10):
    Xtr, ytr, _, _, _ = build("base", TR, h, "onset"); Xva, yva, _, _, _ = build("base", VA, h, "onset"); Xte, yte, rte, _, pos = build("base2", range(40), h, "onset")
    ids = np.unique(rte); idx = {p: np.where(rte == p)[0] for p in ids}
    for nm, fn in models(Xtr, ytr).items():
        sva, ste = fn(Xva), fn(Xte)
        for target in (0.7, 0.8, 0.9):
            t = thr_for_recall(yva, sva, target); pred = ste >= t
            def met(m_):
                y, p = yte[m_], pred[m_]; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn_ = ((~p) & (y == 1)).sum()
                return tp / max(1, tp + fn_), tp / max(1, tp + fp), 100 * fp / len(y)
            pt = met(np.arange(len(yte))); bs = np.array([met(np.concatenate([idx[p] for p in rng.choice(ids, len(ids))])) for _ in range(500)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); ev = event_metrics(ste, pos, h, t)
            ops.append({"h": h, "model": nm, "target": target, "recall": pt[0], "precision": pt[1], "fa": pt[2], "recall_ci": [lo[0], hi[0]], "precision_ci": [lo[1], hi[1]], "fa_ci": [lo[2], hi[2]], "detected": ev["detected"], "events": ev["events"]})
            print(h, nm, target, "rec %.2f prec %.2f FA %.1f [%.1f,%.1f] ev %d/%d" % (pt[0], pt[1], pt[2], lo[2], hi[2], ev["detected"], ev["events"]), flush=True)
json.dump(ops, open("onset_ops.json", "w"), indent=1)
# state task: paired AUC difference versus persistence, Brier score
extra = []
for h in (5, 10):
    Xtr, ytr, _, _, _ = build("base", TR, h, "state"); Xte, yte, rte, cte, _ = build("base2", range(40), h, "state")
    ms = models(Xtr, ytr); ids = np.unique(rte); idx = {p: np.where(rte == p)[0] for p in ids}
    for nm in ("Logistic regression", "Gradient boosting"):
        s = ms[nm](Xte); diffs = []
        for _ in range(500):
            m_ = np.concatenate([idx[p] for p in rng.choice(ids, len(ids))])
            if len(np.unique(yte[m_])) == 2: diffs.append(roc_auc_score(yte[m_], s[m_]) - roc_auc_score(yte[m_], cte[m_]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        extra.append({"h": h, "model": nm, "auc": roc_auc_score(yte, s), "auc_persistence": roc_auc_score(yte, cte), "diff": roc_auc_score(yte, s) - roc_auc_score(yte, cte), "diff_ci": [lo, hi], "brier": brier_score_loss(yte, s), "brier_persistence": brier_score_loss(yte, cte), "brier_base_rate": brier_score_loss(yte, np.full(len(yte), yte.mean()))})
        print(extra[-1], flush=True)
json.dump(extra, open("state_extra.json", "w"), indent=1)
