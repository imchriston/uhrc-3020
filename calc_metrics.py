"""
calc_metrics.py
Reads trial .npz files from eval_uhrc_benchmark.py and prints per-scenario
averages for Success Rate, RMSE, IAE, ITAE, and PLR.

Usage:
  python calc_metrics.py
  python calc_metrics.py --dir benchmark_results/timeseries
"""

import os
import glob
import argparse
import numpy as np
from collections import defaultdict

_TC_IDS = {
    "open_field":        "TC-01",
    "narrow_corridor":   "TC-03",
    "gap_navigation":    "TC-04",
    "dynamic_obstacles": "TC-10",
}

DT = 0.01

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="benchmark_results/timeseries")
    return p.parse_args()


def main():
    args = parse_args()

    npz_files = glob.glob(os.path.join(args.dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {args.dir}")
        return

    metrics = defaultdict(lambda: defaultdict(list))

    for fpath in npz_files:
        stem = os.path.basename(fpath).replace(".npz", "")
        if "_t" not in stem:
            continue
        scenario = stem.rsplit("_t", 1)[0]

        try:
            d       = np.load(fpath)
            dists   = d["dist_log"].astype(float)
            traj    = d["trajectory"]
            start   = d["start"]
            goal    = d["goal"]
            success = bool(d["success"][0])

            t_arr = np.arange(len(dists)) * DT

            def _scalar(key, fallback):
                v = d[key] if key in d else None
                return float(v[0]) if v is not None else fallback

            rmse = _scalar("rmse", float(np.sqrt(np.mean(dists ** 2))))
            iae  = _scalar("iae",  float(_trapz(np.abs(dists), t_arr)))
            itae = _scalar("itae", float(_trapz(t_arr * np.abs(dists), t_arr)))

            path_len = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)))
            straight = float(np.linalg.norm(goal[:2] - start[:2]))
            plr = _scalar("plr", path_len / max(straight, 1e-6))

            metrics[scenario]["rmse"].append(rmse)
            metrics[scenario]["iae"].append(iae)
            metrics[scenario]["itae"].append(itae)
            metrics[scenario]["plr"].append(plr)
            metrics[scenario]["success"].append(success)

        except Exception as e:
            print(f"  Warning: could not read {os.path.basename(fpath)}: {e}")

    col_w = [8, 22, 9, 13, 12, 12, 10]
    hdr   = ["TC-ID", "Scenario", "Succ (%)", "RMSE (m)", "IAE", "ITAE", "PLR"]
    sep   = "─" * (sum(col_w) + 2 * len(col_w))

    print(f"\n{sep}")
    print("  Benchmark Metrics — averaged across trials")
    print(sep)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(hdr, col_w)))
    print(sep)

    order = list(_TC_IDS.keys())
    for scenario in sorted(metrics, key=lambda s: order.index(s) if s in order else 99):
        m  = metrics[scenario]
        tc = _TC_IDS.get(scenario, "??")
        row = [
            tc,
            scenario,
            f"{np.mean(m['success']) * 100:.1f}",
            f"{np.mean(m['rmse']):.3f}",
            f"{np.mean(m['iae']):.3f}",
            f"{np.mean(m['itae']):.3f}",
            f"{np.mean(m['plr']):.3f}",
        ]
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    print(sep + "\n")


if __name__ == "__main__":
    main()