import json
import numpy as np
from sim import simulate

def jain(x):
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))

out = []
for N in (10, 20, 50):
    for G in (0.5, 0.9, 1.5):
        js, top = [], []
        for s in range(5):
            r = simulate(N=N, offered_load=G, L=24, T=200_000, seed=100 + s)
            js.append(jain(r["succ_by"]))
            sb = np.sort(r["succ_by"])[::-1]
            top.append(sb[0] / max(1, sb.sum()))
        row = {"N": N, "G": G, "jain": float(np.mean(js)), "jain_sd": float(np.std(js, ddof=1)),
               "top_share": float(np.mean(top))}
        out.append(row)
        print(row, flush=True)
json.dump(out, open("fairness.json", "w"), indent=1)
