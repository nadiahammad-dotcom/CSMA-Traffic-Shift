"""Generate non-stationary traffic runs: offered load follows a sticky random walk
over discrete levels (changes by at most one level, with prob 0.05 per window)."""
import numpy as np, json
from multiprocessing import Pool
from sim import simulate

import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
CFG = {"base": (20, {}), "n50": (50, {}), "real": (20, {"ifg": 1, "coll_cost": 2}), "burst": (20, {}), "base2": (20, {})}
N, KW = CFG[TAG]
L, W, NW = 24, 500, 600
LEVELS = np.array([0.3, 0.5, 0.7, 0.85, 1.0, 1.15])
RUNS = 40
SEED0 = 100 if TAG == "base2" else 0   # base2: 40 further independent runs of the base process

def make_levels(seed):
    rng = np.random.default_rng(10_000 + seed)
    k = rng.integers(0, len(LEVELS))
    out = np.empty(NW, dtype=int)
    for w in range(NW):
        if rng.random() < 0.05:
            k = min(max(k + rng.choice([-1, 1]), 0), len(LEVELS) - 1)
        out[w] = k
    return out

def make_burst(seed):
    """Alternative traffic: low background load with short, intermittent bursts (mean 6 windows,
    start probability 0.04 per window) at 1.0-1.3 times capacity."""
    rng = np.random.default_rng(20_000 + seed)
    out = np.empty(NW); left = 0; hi = 1.0; base = rng.choice([0.3, 0.5, 0.7])
    for w in range(NW):
        if left == 0 and rng.random() < 0.04:
            left = int(rng.geometric(1 / 6)); hi = rng.uniform(1.0, 1.3)
        if rng.random() < 0.02:
            base = rng.choice([0.3, 0.5, 0.7])
        out[w] = hi if left > 0 else base
        left = max(0, left - 1)
    return out

def one(seed):
    if TAG == "burst":
        G = make_burst(seed); lv = G
        prof = lambda t: float(G[min(t // W, NW - 1)])
    else:
        lv = make_levels(seed)
        prof = lambda t: float(LEVELS[lv[min(t // W, NW - 1)]])
    r = simulate(N=N, offered_load=1.0, L=L, T=NW * W, W=W, seed=seed, load_profile=prof, **KW)
    s = r["series"][:NW]
    arr = np.array([x["thr"] for x in s])
    return seed, lv, s

if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, range(SEED0, SEED0 + RUNS))
    D = {"seed": [], "G": [], "thr": [], "coll": [], "arr": [], "qlen": []}
    for seed, lv, s in res:
        n = len(s)
        D["seed"].append(seed)
        D["G"].append(lv[:n].tolist() if TAG == "burst" else LEVELS[lv[:n]].tolist())
        D["thr"].append([x["thr"] for x in s])
        D["coll"].append([x["coll_rate"] for x in s])
        D["arr"].append([x["arr_rate"] for x in s])
        D["qlen"].append([x["qlen"] for x in s])
    json.dump(D, open("series.json" if TAG == "base" else f"series_{TAG}.json", "w"))
    q = np.concatenate([np.array(x) for x in D["qlen"]])
    print(TAG, "windows", q.size, "frac >=2N:", float((q >= 2 * N).mean()))
