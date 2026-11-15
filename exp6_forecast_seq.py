"""Forecasting with hub-observable features, now including recurrent models (LSTM, GRU).
Usage: python3 exp6_forecast_seq.py <tag>   tag in {base, n50, real}
Splits are by run: 0-23 train, 24-29 validation (threshold + early stopping), 30-39 test."""
import json, sys, warnings, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
N = {"base": 20, "n50": 50, "real": 20}[TAG]
L, K, QSTAR = 24, 8, 2 * N
HORIZONS = [1, 3, 5, 10]
D = json.load(open("series.json" if TAG == "base" else f"series_{TAG}.json"))
TRAIN, VAL, TEST = list(range(0, 24)), list(range(24, 30)), list(range(30, 40))


def feats(run):
    thr, coll = np.array(D["thr"][run]), np.array(D["coll"][run])
    rho = np.array(D["arr"][run]) * L
    q = np.array(D["qlen"][run], float)
    sr = np.maximum(thr / L, 1e-9)
    base = np.stack([thr, coll * 10, np.minimum(coll / sr, 5.0)], 1)      # hub-observable only
    return base, q, rho


def build(idx, h, onset_only):
    X, y, rid, rho_now = [], [], [], []
    for r in idx:
        base, q, rho = feats(r)
        for i in range(K - 1, len(q) - h):
            if onset_only and q[i] >= QSTAR:
                continue
            X.append(base[i - K + 1:i + 1]); y.append(int(q[i + h] >= QSTAR)); rid.append(r); rho_now.append(rho[i])
    return np.array(X), np.array(y), np.array(rid), np.array(rho_now)


def best_thr(y, s):
    cs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    f = [f1_score(y, s >= c, zero_division=0) for c in cs]
    return cs[int(np.argmax(f))]


def boot_auc(y, s, rid, B=300, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(rid); vals = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids)); m = np.concatenate([np.where(rid == p)[0] for p in pick])
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def seq_model(kind):
    tf.keras.utils.set_random_seed(0)
    layer = tf.keras.layers.LSTM if kind == "LSTM" else tf.keras.layers.GRU
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(K, 3)), layer(32), tf.keras.layers.Dense(16, activation="relu"),
                             tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def fit_seq(kind, Xtr, ytr, Xva, yva):
    mu, sd = Xtr.reshape(-1, 3).mean(0), Xtr.reshape(-1, 3).std(0) + 1e-9
    f = lambda A: (A - mu) / sd
    m = seq_model(kind)
    w = {0: 1.0, 1: float(min(20.0, np.sqrt((len(ytr) - ytr.sum()) / max(1, ytr.sum()))))}
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    m.fit(f(Xtr), ytr, validation_data=(f(Xva), yva), epochs=25, batch_size=128, class_weight=w, callbacks=[es], verbose=0)
    return lambda A: m.predict(f(A), verbose=0).ravel()


results = []
for onset in (False, True):
    for h in HORIZONS:
        Xtr, ytr, rtr, _ = build(TRAIN, h, onset); Xva, yva, rva, rhova = build(VAL, h, onset); Xte, yte, rte, rhote = build(TEST, h, onset)
        flat = lambda A: A.reshape(len(A), -1)
        scorers = {"Oracle load rule": (rhova, rhote),
                   "Utilisation rule": (Xva[:, -1, 0], Xte[:, -1, 0])}
        for name, mk in {"Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                         "GradBoost": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
                         "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=400, random_state=0))}.items():
            m = mk().fit(flat(Xtr), ytr)
            scorers[name] = (m.predict_proba(flat(Xva))[:, 1], m.predict_proba(flat(Xte))[:, 1])
        for kind in ("LSTM", "GRU"):
            pred = fit_seq(kind, Xtr, ytr, Xva, yva)
            scorers[kind] = (pred(Xva), pred(Xte))
        row = {"tag": TAG, "task": "onset" if onset else "state", "h": h, "n_test": int(len(yte)), "n_pos_test": int(yte.sum())}
        for name, (sva, ste) in scorers.items():
            thr = 0.85 if name in ("Oracle load rule", "Utilisation rule") else best_thr(yva, sva)
            pred_lab = ste >= thr; lo, hi = boot_auc(yte, ste, rte)
            row[name] = {"auc": float(roc_auc_score(yte, ste)), "auc_lo": lo, "auc_hi": hi,
                         "f1": float(f1_score(yte, pred_lab, zero_division=0)),
                         "prec": float(precision_score(yte, pred_lab, zero_division=0)),
                         "rec": float(recall_score(yte, pred_lab, zero_division=0)), "thr": float(thr)}
        results.append(row)
        print(TAG, row["task"], "h", h, "npos", row["n_pos_test"],
              {k: (round(v["auc"], 3), round(v["f1"], 2)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
json.dump(results, open(f"forecast_seq_{TAG}.json", "w"), indent=1)
