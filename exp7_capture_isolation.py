"""Capture isolation. cw_mult = 0: BEB. cw_mult = m > 0: fixed window m*N after every collision (no collision history).
cw_mult = 99: BEB with winner keeping half its exponent. cw_mult = -m: as above and the winner also draws a random backoff before its next frame (no capture)."""
import json, numpy as np
from multiprocessing import Pool
from sim import simulate
L, T = 24, 200_000
def jain(x):
    x = np.asarray(x, float); return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))
def job(a):
    N, G, cwmul, s = a
    mem = cwmul == 99                       # 99: standard BEB, but the winner keeps half of its backoff exponent
    pb = cwmul < 0
    cw = abs(cwmul) * N if cwmul and not mem else None
    r = simulate(N=N, offered_load=G, L=L, T=T, seed=400 + s, fixed_cw=cw, post_backoff=pb, memory=mem)
    return N, G, cwmul, jain(r["succ_by"]), r["throughput"], max(r["succ_by"]) / max(1, sum(r["succ_by"]))
if __name__ == "__main__":
    jobs = [(N, G, c, s) for N in (10, 20, 50) for G in (0.9, 1.5) for c in (0, 2, -2, -4, 99) for s in range(10)]
    with Pool(2) as p: res = p.map(job, jobs)
    agg = {}
    for N, G, c, j, thr, top in res: agg.setdefault((N, G, c), []).append((j, thr, top * N))
    out = []
    for k, v in sorted(agg.items()):
        a = np.array(v); m, sd = a.mean(0), a.std(0, ddof=1)
        out.append({"N": k[0], "G": k[1], "cw_mult": k[2], "jain": m[0], "jain_sd": sd[0], "thr": m[1], "top_x_fair": m[2]})
        print(k, "J %.3f±%.3f thr %.3f top/fair %.2f" % (m[0], sd[0], m[1], m[2]), flush=True)
    json.dump(out, open("capture_isolation.json", "w"), indent=1)
