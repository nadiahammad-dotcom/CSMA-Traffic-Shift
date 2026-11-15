import json, sys
import numpy as np
from sim import simulate

Ns = [10, 20, 50]
loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
seeds = list(range(5))
L = 24
T = 200_000

rows = []
for N in Ns:
    for G in loads:
        res = [simulate(N=N, offered_load=G, L=L, T=T, seed=s) for s in seeds]
        def agg(k):
            v = np.array([r[k] for r in res], dtype=float)
            return float(v.mean()), float(v.std(ddof=1))
        row = {"N": N, "G": G}
        for k in ["throughput", "p_coll_per_event", "mean_delay_slots",
                  "p95_delay_slots", "drop_rate", "final_queue"]:
            m, s = agg(k)
            row[k] = m
            row[k + "_sd"] = s
        rows.append(row)
        print(N, G, round(row["throughput"], 3), round(row["p_coll_per_event"], 3),
              round(row["mean_delay_slots"], 1), round(row["drop_rate"], 4), flush=True)

json.dump(rows, open("sweep.json", "w"), indent=1)
