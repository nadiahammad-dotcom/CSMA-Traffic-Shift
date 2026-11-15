"""
Slotted shared-medium Ethernet (hub) simulation with binary exponential backoff.

Model (stated plainly so the paper can describe it exactly):
- N stations share one collision domain (a repeater hub).
- Time is slotted; one slot = 512 bit times (Ethernet slot time).
- A frame transmission occupies L slots (frame length in slots).
- Each station has an infinite FIFO queue; frames arrive as a Bernoulli process (at most one per slot, discrete-time analogue of Poisson) with
  per-station rate lam (frames per slot).
- A station with a head-of-line frame and backoff counter 0 transmits in the slot.
  If exactly one station transmits, success: channel busy for L slots.
  If two or more transmit, collision: 1 slot is lost, each colliding station
  draws backoff uniformly from {0..2^min(k,10)-1}; frame dropped after 16 attempts.
- Carrier sense: stations defer while the channel is busy (1-persistent behaviour
  is approximated by letting backoff counters freeze during busy periods).

Outputs per run: throughput (fraction of slots carrying successful payload),
collision probability per transmission attempt, mean access delay, drop rate,
and a per-window time series (window = W slots) for forecasting experiments.
"""
import numpy as np


def simulate(N=20, offered_load=0.5, L=24, T=200_000, W=500, seed=0,
             max_attempts=16, max_exp=10, load_profile=None,
             coll_cost=1, ifg=0, memory=False, fixed_cw=None, post_backoff=False, poisson=False):
    # poisson: arrivals per station and slot are Poisson(lam) instead of Bernoulli(lam) (robustness check; default off)
    # post_backoff: with fixed_cw, the winner also draws a random backoff before its next frame
    # fixed_cw: capture-isolation control; every collision draws backoff uniformly from {0..fixed_cw-1},
    # so no station carries a backoff history (winner and losers are statistically identical)
    # coll_cost: slots lost per collision (1 = ideal; >1 adds jam/propagation cost)
    # ifg: extra idle slots after every successful frame (interframe gap)
    # memory: capture-control variant; after a success the winner keeps half of its backoff exponent
    rng = np.random.default_rng(seed)
    # offered_load G = N * lam * L  (fraction of capacity offered)
    lam = offered_load / (N * L)
    queues = [[] for _ in range(N)]          # arrival slot of each queued frame
    attempts = np.zeros(N, dtype=int)
    backoff = np.zeros(N, dtype=int)
    busy_until = 0
    succ = coll_events = tx_attempts = drops = 0
    succ_by = np.zeros(N, dtype=int)
    delays = []
    win = {"succ": 0, "coll": 0, "att": 0, "arr": 0, "qsum": 0.0}
    series = []
    t = 0
    while t < T:
        cur_lam = lam if load_profile is None else lam * load_profile(t)
        arr = rng.poisson(cur_lam, N) if poisson else (rng.random(N) < cur_lam).astype(int)
        for i in np.nonzero(arr)[0]:
            queues[i].extend([t] * int(arr[i]))
        win["arr"] += int(arr.sum())
        if t >= busy_until:
            ready = [i for i in range(N) if queues[i] and backoff[i] == 0]
            if len(ready) == 1:
                i = ready[0]
                tx_attempts += 1
                win["att"] += 1
                arrival = queues[i].pop(0)
                delays.append(t - arrival)
                if memory:
                    attempts[i] = attempts[i] // 2
                    backoff[i] = rng.integers(0, 2 ** min(attempts[i], max_exp)) if attempts[i] > 0 else 0
                else:
                    attempts[i] = 0
                    if post_backoff and fixed_cw:
                        backoff[i] = rng.integers(0, fixed_cw)
                succ += 1
                succ_by[i] += 1
                win["succ"] += 1
                busy_until = t + L + ifg
                t_next = t + L + ifg
                # advance time by L slots, accumulate queue samples
                # slots t+1 .. t+L-1 are generated here; slot t+L is generated
                # by the next loop iteration (avoids double-counting arrivals)
                for step in range(L + ifg - 1):
                    tt = t + 1 + step
                    if tt >= T:
                        break
                    a2 = rng.poisson(cur_lam, N) if poisson else (rng.random(N) < cur_lam).astype(int)
                    for j in np.nonzero(a2)[0]:
                        queues[j].extend([tt] * int(a2[j]))
                    win["arr"] += int(a2.sum())
                t = t_next
            elif len(ready) >= 2:
                coll_events += 1
                win["coll"] += 1
                tx_attempts += len(ready)
                win["att"] += len(ready)
                for i in ready:
                    attempts[i] += 1
                    if attempts[i] >= max_attempts:
                        queues[i].pop(0)
                        drops += 1
                        attempts[i] = 0
                        backoff[i] = 0
                    else:
                        k = min(attempts[i], max_exp)
                        backoff[i] = rng.integers(0, fixed_cw if fixed_cw else 2 ** k)
                for step in range(coll_cost - 1):
                    tt = t + 1 + step
                    if tt >= T:
                        break
                    a2 = rng.poisson(cur_lam, N) if poisson else (rng.random(N) < cur_lam).astype(int)
                    for j in np.nonzero(a2)[0]:
                        queues[j].extend([tt] * int(a2[j]))
                    win["arr"] += int(a2.sum())
                t += coll_cost
            else:
                for i in range(N):
                    if queues[i] and backoff[i] > 0:
                        backoff[i] -= 1
                t += 1
        else:
            t += 1
        if t % W < 1 or (t // W) > len(series):
            # close window(s)
            while (t // W) > len(series):
                qlen = sum(len(q) for q in queues)
                series.append({
                    "thr": win["succ"] * L / W,
                    "coll_rate": win["coll"] / W,
                    "arr_rate": win["arr"] / W,
                    "qlen": qlen,
                    "att": win["att"],
                })
                win = {"succ": 0, "coll": 0, "att": 0, "arr": 0, "qsum": 0.0}
    thr = succ * L / T
    pcoll = (coll_events / max(1, (succ + coll_events)))
    return {
        "throughput": thr,
        "p_coll_per_event": pcoll,
        "collisions": coll_events,
        "successes": succ,
        "mean_delay_slots": float(np.mean(delays)) if delays else float("nan"),
        "p95_delay_slots": float(np.percentile(delays, 95)) if delays else float("nan"),
        "drops": drops,
        "drop_rate": drops / max(1, (succ + drops)),
        "final_queue": sum(len(q) for q in queues),
        "succ_by": succ_by.tolist(),
        "series": series,
    }


if __name__ == "__main__":
    r = simulate(N=20, offered_load=0.5, T=50_000, seed=1)
    print({k: v for k, v in r.items() if k != "series"})
