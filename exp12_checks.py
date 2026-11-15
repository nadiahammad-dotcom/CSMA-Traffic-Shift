"""Model checks: (1) Bernoulli versus Poisson arrivals; (2) queue growth over a run five times longer. Output: model_checks.json"""
import json, numpy as np
from multiprocessing import Pool
from sim import simulate
def job(a):
    G, pois, s = a
    r = simulate(N=20, offered_load=G, T=200_000, seed=500 + s, poisson=pois)
    return G, pois, r["throughput"], r["mean_delay_slots"], r["p_coll_per_event"]
def long(T):
    r = simulate(N=20, offered_load=1.2, T=T, seed=0)
    return T, r["final_queue"], r["throughput"]
if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(job, [(G, pb, s) for G in (0.8, 0.9, 1.0, 1.5) for pb in (False, True) for s in range(5)])
        lg = p.map(long, [200_000, 1_000_000])
    out = {"arrivals": [], "long_run": [{"T": T, "final_queue": q, "throughput": t} for T, q, t in lg]}
    for G in (0.8, 0.9, 1.0, 1.5):
        row = {"G": G}
        for pb, nm in ((False, "bernoulli"), (True, "poisson")):
            a = np.array([r[2:] for r in res if r[0] == G and r[1] == pb]); row[nm] = {"throughput": a[:, 0].mean(), "delay": a[:, 1].mean(), "coll": a[:, 2].mean(), "thr_sd": a[:, 0].std(ddof=1)}
        out["arrivals"].append(row); print(row, flush=True)
    print(out["long_run"])
    json.dump(out, open("model_checks.json", "w"), indent=1)
