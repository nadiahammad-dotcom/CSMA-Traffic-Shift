"""Robustness of the strict early-warning evaluation to station count and overhead (tags n50, real).
Usage: python3 exp10_strict_robust.py <tag>.  Same definitions as exp9_strict.py; train 0-23, validation 24-29, test 30-39.
Strict early-warning evaluation, persistence baselines and cross-traffic transfer.
Usage: python3 exp9_strict.py
Scenarios (train/validation runs always come from the training traffic):
  A  train base (runs 0-23, val 24-29)  -> test base (runs 30-39) and base2 (40 new runs)
  B  train base                         -> test burst (40 runs, a different traffic process)
  C  train burst (runs 0-23, val 24-29) -> test burst (30-39) and base (all 40)
Tasks (Q* = 2N frames of backlog; windows of 500 slots; last K = 8 windows of hub-visible features):
  state  : congested at t+h (all windows).  Extra baseline: persistence of the true state (needs backlog).
  onset  : among windows uncongested at t, at least one congested window in t+1..t+h (first crossing within h).
Event metrics for onset: an event is an uncongested -> congested transition at window c; it is detected if the
score exceeded the threshold in some window of [c-h, c-1]; false alarms are alarms raised in uncongested
windows that are not followed by a crossing within h windows."""
import json, os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

import sys
TAG = sys.argv[1]
N = 50 if TAG == "n50" else 20
L, K = 24, 8
QSTAR = 2 * N
HS = [5, 10]
SETS = {TAG: json.load(open(f"series_{TAG}.json"))}


def run_arrays(D, r):
    thr, coll = np.array(D["thr"][r]), np.array(D["coll"][r])
    sr = np.maximum(thr / L, 1e-9)
    feat = np.stack([thr, coll * 10, np.minimum(coll / sr, 5.0)], 1)
    return feat, np.array(D["qlen"][r], float)


def build(name, runs, h, task):
    D = SETS[name]; X, y, rid, cur, pos = [], [], [], [], []
    for r in runs:
        f, q = run_arrays(D, r)
        for i in range(K - 1, len(q) - h):
            if task == "onset" and q[i] >= QSTAR:
                continue
            if task == "state":
                lab = int(q[i + h] >= QSTAR)
            else:
                lab = int((q[i + 1:i + h + 1] >= QSTAR).any())
            X.append(f[i - K + 1:i + 1]); y.append(lab); rid.append(f"{name}{r}"); cur.append(float(q[i] >= QSTAR)); pos.append((name, r, i))
    return np.array(X), np.array(y), np.array(rid), np.array(cur), pos


def boot_auc(y, s, rid, B=300, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(rid); v = []
    idx = {p: np.where(rid == p)[0] for p in ids}
    for _ in range(B):
        m = np.concatenate([idx[p] for p in rng.choice(ids, len(ids))])
        if len(np.unique(y[m])) == 2:
            v.append(roc_auc_score(y[m], s[m]))
    return np.percentile(v, [2.5, 97.5]).tolist()


def thr_for_recall(y, s, target=0.8):
    cs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    ok = [c for c in cs if (s[y == 1] >= c).mean() >= target]
    return max(ok) if ok else cs[0]


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return p, r, (2 * p * r / max(1e-9, p + r)), fp


def event_metrics(score, pos, h, thr):
    """event-level detection rate, median lead (windows) and false alarms per 100 uncongested windows"""
    by = {}
    for sc, (name, r, i) in zip(score, pos):
        by.setdefault((name, r), {})[i] = sc
    ev = det = 0; leads = []; fa = n_unc = 0
    for (name, r), d in by.items():
        _, q = run_arrays(SETS[name], r)
        for c in range(1, len(q)):
            if q[c] >= QSTAR and q[c - 1] < QSTAR:
                ev += 1
                al = [t for t in range(c - h, c) if t in d and d[t] >= thr]
                if al:
                    det += 1; leads.append(c - min(al))
        for i, sc in d.items():
            n_unc += 1
            if sc >= thr and not (q[i + 1:i + h + 1] >= QSTAR).any():
                fa += 1
    return {"events": ev, "detected": det, "median_lead": float(np.median(leads)) if leads else None,
            "false_alarm_per_100": 100 * fa / max(1, n_unc)}


def fit_lstm(Xtr, ytr, Xva, yva):
    tf.keras.utils.set_random_seed(0)
    mu, sd = Xtr.reshape(-1, 3).mean(0), Xtr.reshape(-1, 3).std(0) + 1e-9
    f = lambda A: (A - mu) / sd
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(K, 3)), tf.keras.layers.LSTM(32),
                             tf.keras.layers.Dense(16, activation="relu"), tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    w = {0: 1.0, 1: float(min(20.0, np.sqrt((len(ytr) - ytr.sum()) / max(1, ytr.sum()))))}
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    m.fit(f(Xtr), ytr, validation_data=(f(Xva), yva), epochs=25, batch_size=128, class_weight=w, callbacks=[es], verbose=0)
    return lambda A: m.predict(f(A), verbose=0).ravel()


SCEN = {"R": (TAG, {TAG: range(30, 40)})}
TR, VA = range(0, 24), range(24, 30)
out = []
for sc, (trn, tests) in SCEN.items():
    for task in ("state", "onset"):
        for h in HS:
            Xtr, ytr, _, _, _ = build(trn, TR, h, task); Xva, yva, _, cva, _ = build(trn, VA, h, task)
            flat = lambda A: A.reshape(len(A), -1)
            models = {}
            for nm, mk, fe in (("Logistic (8 windows)", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), flat),
                               ("Logistic (last window)", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), lambda A: A[:, -1, :]),
                               ("GradBoost", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0), flat)):
                m = mk().fit(fe(Xtr), ytr); models[nm] = (lambda m=m, fe=fe: (lambda A: m.predict_proba(fe(A))[:, 1]))()
            models["LSTM"] = fit_lstm(Xtr, ytr, Xva, yva)
            models["Utilisation rule"] = lambda A: A[:, -1, 0]
            for tname, runs in tests.items():
                Xte, yte, rte, cte, pos = build(tname, runs, h, task)
                row = {"scenario": sc, "train": trn, "test": tname, "task": task, "h": h, "n": int(len(yte)), "n_pos": int(yte.sum())}
                preds = dict(models)
                for nm, fn in preds.items():
                    s_te, s_va = fn(Xte), fn(Xva)
                    d = {"auc": float(roc_auc_score(yte, s_te)), "ap": float(average_precision_score(yte, s_te))}
                    d["auc_lo"], d["auc_hi"] = boot_auc(yte, s_te, rte)
                    if task == "onset":
                        t80 = thr_for_recall(yva, s_va, 0.8)
                        p, r, f1, fp = prf(yte, (s_te >= t80).astype(int))
                        d.update({"thr80": float(t80), "prec80": p, "rec80": r, "f1_80": f1})
                        d["events"] = event_metrics(s_te, pos, h, t80)
                    row[nm] = d
                if task == "state":
                    p = cte  # persistence of the true (unobservable) current state
                    row["Persistence (needs backlog)"] = {"auc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p))}
                row["base_rate"] = float(yte.mean())
                out.append(row)
                print(sc, tname, task, h, "pos", row["n_pos"], "/", row["n"],
                      {k: round(v["auc"], 3) for k, v in row.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open(f"strict_eval_{TAG}.json", "w"), indent=1)
