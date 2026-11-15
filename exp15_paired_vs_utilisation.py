TR, VA = range(0, 24), range(24, 30)
B = 2000
out = []
for task in ("state", "onset"):
    for h in HS:
        Xtr, ytr, _, _, _ = build("base", TR, h, task)
        Xte, yte, rte, _, _ = build("base2", range(40), h, task)
        flat = lambda A: A.reshape(len(A), -1)
        m8 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(flat(Xtr), ytr)
        m1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[:, -1, :], ytr)
        sc = {"Logistic (8 windows)": m8.predict_proba(flat(Xte))[:, 1],
              "Logistic (last window)": m1.predict_proba(Xte[:, -1, :])[:, 1],
              "Utilisation rule": Xte[:, -1, 0]}
        ids = np.unique(rte); idx = {p: np.where(rte == p)[0] for p in ids}
        rng = np.random.default_rng(7)
        diffs = {k: [] for k in ("Utilisation rule", "Logistic (last window)")}
        for _ in range(B):
            mm = np.concatenate([idx[p] for p in rng.choice(ids, len(ids))])
            if len(np.unique(yte[mm])) < 2: continue
            a = roc_auc_score(yte[mm], sc["Logistic (8 windows)"][mm])
            for k in diffs: diffs[k].append(a - roc_auc_score(yte[mm], sc[k][mm]))
        row = {"task": task, "h": h, "auc": {k: float(roc_auc_score(yte, v)) for k, v in sc.items()}, "B": len(diffs["Utilisation rule"])}
        for k, v in diffs.items():
            v = np.array(v)
            row["diff_vs " + k] = {"diff": float(roc_auc_score(yte, sc["Logistic (8 windows)"]) - roc_auc_score(yte, sc[k])),
                                   "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                                   "share_le_0": float((v <= 0).mean())}
        out.append(row); print(row, flush=True)
json.dump(out, open("paired_vs_utilisation.json", "w"), indent=1)
