"""Capture control and realism sensitivity.
Variants: base (ideal), memory (capture control), real (ifg=1 slot, coll_cost=2 slots)."""
import json, numpy as np
from multiprocessing import Pool
from sim import simulate

L, T = 24, 200_000
VARIANTS = {"base": {}, "memory": {"memory": True}, "real": {"ifg": 1, "coll_cost": 2}}
SEEDS = range(300, 305)

def jain(x):
    x = np.asarray(x, float); return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))

def job(a):
    N, G, v, s = a
    r = simulate(N=N, offered_load=G, L=L, T=T, seed=s, **VARIANTS[v])
    sb = np.sort(r["succ_by"])[::-1]
    return (N, G, v, s, r["throughput"], jain(r["succ_by"]), float(sb[0] / max(1, sb.sum())),
            r["mean_delay_slots"], r["p_coll_per_event"])

if __name__ == "__main__":
    jobs = []
    for v in VARIANTS:
        for N in (10, 20, 50):
            for G in ((0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5) if N == 20 else (0.9, 1.5)):
                for s in SEEDS: jobs.append((N, G, v, s))
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    agg = {}
    for N, G, v, s, thr, j, top, dly, pc in res:
        agg.setdefault((N, G, v), []).append((thr, j, top, dly, pc))
    out = []
    for (N, G, v), rows in sorted(agg.items()):
        a = np.array(rows); m, sd = a.mean(0), a.std(0, ddof=1)
        out.append({"N": N, "G": G, "variant": v, "thr": m[0], "thr_sd": sd[0], "jain": m[1], "jain_sd": sd[1],
                    "top": m[2], "delay": m[3], "delay_sd": sd[3], "pcoll": m[4]})
        print(N, G, v, "thr %.3f J %.3f top*N %.2f dly %.0f" % (m[0], m[1], m[2] * N, m[3]), flush=True)
    json.dump(out, open("capture_realism.json", "w"), indent=1)
