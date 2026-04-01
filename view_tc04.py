"""
view TC04 dynamic obstacles testcases

Panels produced:
  Window 1 — XY trajectory + dynamic obstacle positions + metric box
  Window 2 — Distance-to-goal time series + appearance markers + RMSE/IAE/ITAE
  Window 3 — Control inputs Fz, τφ, τθ, τψ over time

Metrics displayed:
  RMSE  root-mean-square distance to goal          (m)
  IAE   integral absolute error                    (m·s)
  ITAE  integral time-weighted absolute error      (m·s²)
  PLR   path length / straight-line distance       


"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # swap to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

TITLE_FS  = 14
LABEL_FS  = 13
TICK_FS   = 11
LEGEND_FS = 11
ANNOT_FS  = 10
METRIC_FS = 10

BG         = "#FFFFFF"
PANEL_BG   = "#F7F8FA"
GRID_COL   = "#DADDE3"
OBS_FACE   = "#D6303080"
OBS_EDGE   = "#A31515"
DYN_EDGE   = "#7B1FA2"          # dynamic obstacle outline
DYN_FACE   = "#CE93D880"        # dynamic obstacle fill (ghost)
DYN_ACTIVE = "#9C27B0"          # dynamic obstacle trail / final pos
GOAL_FACE  = "#1E8B4420"
GOAL_EDGE  = "#1E8B44"
TEXT_MAIN  = "#1A1A2E"
TEXT_SUB   = "#555F70"
ACCENT     = "#2563EB"          # success trajectory
DANGER     = "#DC2626"          # collision
WARN       = "#D97706"          # timeout
HOVER_REF  = "#6B7280"

ARENA        = (-10.0, 10.0)
GOAL_RADIUS  = 0.5
NEAR_MISS_TH = 0.3
GRID_RES     = 0.1
DT           = 0.01

TC_ID    = "TC-D05"
SCENARIO = "dyn_random_pop"

TRAJ_COLOR = {"success": ACCENT, "collision": DANGER, "timeout": WARN}


def _try(d, key, default=None):
    try:
        return d[key]
    except KeyError:
        return default


def load_trial(npz_path: str) -> dict:
    """Load a dyn_random_pop .npz file and compute all display metrics."""
    d = np.load(npz_path, allow_pickle=False)

    traj     = d["trajectory"]          # [T+1, 3]
    start    = d["start"]               # [3]
    goal     = d["goal"]                # [3]
    dist_log = d["dist_log"]            # [T]

    circ_arr = d["circles"]             # [n_static, 3]: cx, cy, r
    circles  = [((float(circ_arr[i, 0]), float(circ_arr[i, 1]), 0.0),
                  float(circ_arr[i, 2]))
                for i in range(len(circ_arr))]

    # Dynamic obstacles
    dyn_arr  = _try(d, "dynamic_obs")  # [n_dyn, 4]: cx0,cy0,r,appear_step
    dyn_traj = _try(d, "dynamic_traj") # [T, n_dyn, 2]  — positions over time

    dyn_obs = []
    if dyn_arr is not None:
        for i in range(len(dyn_arr)):
            dyn_obs.append({
                "cx0":         float(dyn_arr[i, 0]),
                "cy0":         float(dyn_arr[i, 1]),
                "r":           float(dyn_arr[i, 2]),
                "appear_step": int(dyn_arr[i, 3]),
                "trail":       (dyn_traj[:, i, :] if dyn_traj is not None
                                and dyn_traj.shape[1] > i else None),
            })

    final_dist = float(np.linalg.norm(traj[-1, :2] - goal[:2]))
    success    = bool(_try(d, "success",   np.array([final_dist < GOAL_RADIUS]))[0])
    collision  = bool(_try(d, "collision", np.array([False]))[0])
    dyn_hit    = bool(_try(d, "dynamic_obs_hit", np.array([False]))[0])
    timeout    = not success and not collision

    def _scalar(key, fallback):
        v = _try(d, key)
        return float(v[0]) if v is not None else fallback

    dist_arr = dist_log.astype(float)
    t_arr    = np.arange(len(dist_arr)) * DT

    rmse = _scalar("rmse",  float(np.sqrt(np.mean(dist_arr ** 2))))
    iae  = _scalar("iae",   float(_trapz(np.abs(dist_arr), t_arr)))
    itae = _scalar("itae",  float(_trapz(t_arr * np.abs(dist_arr), t_arr)))
    plr  = _scalar("plr",   None)
    if plr is None:
        diffs  = np.diff(traj[:, :2], axis=0)
        plen   = float(np.sum(np.linalg.norm(diffs, axis=1)))
        strait = float(np.linalg.norm(goal[:2] - start[:2]))
        plr    = plen / max(strait, 1e-6)

    near_miss_count   = int(_try(d, "near_miss_count",
                                  np.array([0]))[0])
    reaction_step     = int(_try(d, "reaction_step",
                                  np.array([-1]))[0])
    first_appear_step = int(_try(d, "first_appear_step",
                                  np.array([-1]))[0])

    stem  = Path(npz_path).stem
    parts = stem.rsplit("_trial", 1)
    trial_idx = int(parts[1]) if len(parts) == 2 else 0

    return {
        "trajectory":        traj,
        "start":             start,
        "goal":              goal,
        "dist_log":          dist_log,
        "circles":           circles,
        "dyn_obs":           dyn_obs,
        "fz_log":            _try(d, "fz_log"),
        "tau_phi_log":       _try(d, "tau_phi_log"),
        "tau_theta_log":     _try(d, "tau_theta_log"),
        "tau_psi_log":       _try(d, "tau_psi_log"),
        "z_log":             _try(d, "z_log"),
        "success":           success,
        "collision":         collision,
        "dyn_hit":           dyn_hit,
        "timeout":           timeout,
        "final_dist":        final_dist,
        "trial_idx":         trial_idx,
        "near_miss_count":   near_miss_count,
        "reaction_step":     reaction_step,
        "first_appear_step": first_appear_step,
        "metrics": {
            "rmse": rmse,
            "iae":  iae,
            "itae": itae,
            "plr":  plr,
        },
    }



def _metric_box(metrics: dict) -> str:
    return (f"RMSE : {metrics['rmse']:.3f} m\n"
            f"IAE  : {metrics['iae']:.3f} m·s\n"
            f"ITAE : {metrics['itae']:.3f} m·s²\n"
            f"PLR  : {metrics['plr']:.3f}")



def plot_environment(trial: dict, caption: bool = True):
    circles  = trial["circles"]
    dyn_obs  = trial["dyn_obs"]
    traj     = trial["trajectory"]
    start    = trial["start"]
    goal     = trial["goal"]
    metrics  = trial["metrics"]

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    if circles:
        lo, hi = ARENA
        xs  = np.arange(lo, hi + GRID_RES, GRID_RES)
        ys  = np.arange(lo, hi + GRID_RES, GRID_RES)
        xg, yg = np.meshgrid(xs, ys)
        dist_g = np.full_like(xg, 20.0)
        for (cx, cy, _), r in circles:
            dist_g = np.minimum(dist_g,
                                np.sqrt((xg - cx)**2 + (yg - cy)**2) - r)
        dist_g = np.clip(dist_g, 0.0, 5.0)
        cmap_clr = LinearSegmentedColormap.from_list(
            "clr", ["#C8D6E5", "#EEF2F7", PANEL_BG])
        ax.imshow(dist_g, extent=[-10, 10, -10, 10], origin="lower",
                  cmap=cmap_clr, alpha=0.65, vmin=0, vmax=5,
                  aspect="equal", zorder=0)

    for (cx, cy, _), r in circles:
        ax.add_patch(Circle((cx, cy), r, lw=1.8,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))

    # ── Dynamic obstacles ─────────────────────────────────────────────────────
    for i, obs in enumerate(dyn_obs):
        cx0, cy0, r = obs["cx0"], obs["cy0"], obs["r"]
        appear      = obs["appear_step"]
        trail       = obs["trail"]     # [T, 2] or None

        ax.add_patch(Circle((cx0, cy0), r, lw=1.8, ls="--",
                            edgecolor=DYN_EDGE, facecolor=DYN_FACE,
                            zorder=3))
        ax.text(cx0, cy0,
                f"D{i}\n@s{appear}",
                ha="center", va="center",
                fontsize=ANNOT_FS - 2, color="#4A148C",
                fontweight="bold", zorder=5)

        if trail is not None and appear < len(trail):
            ax.plot(trail[appear:, 0], trail[appear:, 1],
                    color=DYN_ACTIVE, lw=1.2, ls=":", alpha=0.6,
                    zorder=3)
            fx, fy = trail[-1]
            ax.add_patch(Circle((fx, fy), r, lw=1.5,
                                edgecolor=DYN_EDGE,
                                facecolor="#E1BEE7",
                                alpha=0.55, zorder=3))

    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=True,
                        facecolor=GOAL_FACE, zorder=4))
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=False,
                        ls="--", lw=1.5, edgecolor=GOAL_EDGE, zorder=4))

    if len(traj) > 1:
        for i in range(len(traj) - 1):
            alpha = 0.20 + 0.80 * (i / max(len(traj) - 2, 1))
            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                    color=tcolor, lw=2.2, alpha=alpha,
                    solid_capstyle="round", zorder=6)

    # Direction arrows every ~15 steps
    step_a = max(1, len(traj) // 15)
    for i in range(0, len(traj) - 2, step_a):
        dx  = traj[i+1, 0] - traj[i, 0]
        dy  = traj[i+1, 1] - traj[i, 1]
        mag = math.sqrt(dx**2 + dy**2) + 1e-9
        ax.annotate("",
                    xy=(traj[i, 0] + dx/mag*0.55,
                        traj[i, 1] + dy/mag*0.55),
                    xytext=(traj[i, 0], traj[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=tcolor,
                                   lw=1.1, alpha=0.55),
                    zorder=7)

    # Start / end markers
    ax.plot(*start[:2], "o", ms=12, color="#16A34A",
            mec="white", mew=1.5, zorder=8)
    ax.plot(*goal[:2],  "*", ms=17, color="#F59E0B",
            mec="white", mew=1.0, zorder=8)
    ax.plot(*traj[-1, :2], "s", ms=11, color=tcolor,
            mec="white", mew=1.2, zorder=8)

    def fmt(v):
        return (f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
                if len(v) >= 3 else f"({v[0]:.2f}, {v[1]:.2f})")

    coord_text = (f"Start :  {fmt(start)}\n"
                  f"Goal  :  {fmt(goal)}\n"
                  f"Final :  {fmt(traj[-1])}")
    ax.text(0.02, 0.02, coord_text,
            transform=ax.transAxes,
            fontsize=ANNOT_FS, color=TEXT_MAIN, family="monospace",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92),
            zorder=9)

    ax.text(0.98, 0.98, _metric_box(metrics),
            transform=ax.transAxes,
            fontsize=METRIC_FS, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92),
            zorder=9)

    appear_steps = ", ".join(str(o["appear_step"]) for o in dyn_obs)
    dyn_text = (f"Dynamic obs : {len(dyn_obs)}\n"
                f"Appear steps: {appear_steps}\n"
                f"Near-misses : {trial['near_miss_count']} steps\n"
                f"Reaction    : "
                + (f"step {trial['reaction_step']}"
                   if trial['reaction_step'] >= 0 else "not detected"))
    ax.text(0.02, 0.98, dyn_text,
            transform=ax.transAxes,
            fontsize=ANNOT_FS - 1, color=TEXT_MAIN, family="monospace",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=DYN_EDGE, alpha=0.88),
            zorder=9)

    ax.set_xlabel("$x$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("$y$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_xlim(ARENA); ax.set_ylim(ARENA)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    col_label = ("Dynamic obstacle hit" if trial["dyn_hit"]
                 else "Static obstacle hit" if trial["collision"]
                 else outcome.capitalize())
    handles = [
        mpatches.Patch(color=tcolor, alpha=0.85,
                       label=f"{col_label}  (d={trial['final_dist']:.2f} m)"),
        mpatches.Patch(color=OBS_EDGE, alpha=0.6, label="Static obstacle"),
        mpatches.Patch(color=DYN_EDGE, alpha=0.6, label="Dynamic obstacle (pop-in)"),
        mpatches.Patch(color=GOAL_EDGE, alpha=0.6,
                       label=f"Goal zone  (r={GOAL_RADIUS} m)"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#16A34A", ms=11, label="Start"),
        plt.Line2D([0],[0], marker="*", color="w",
                   markerfacecolor="#F59E0B", ms=13, label="Goal"),
        plt.Line2D([0],[0], marker="s", color="w",
                   markerfacecolor=tcolor, ms=10, label="Final position"),
    ]
    leg = ax.legend(handles=handles, loc="lower right",
                    fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    if caption:
        out = ("Success" if trial["success"]
               else "Collision (dynamic)" if trial["dyn_hit"]
               else "Collision (static)" if trial["collision"]
               else "Timeout")
        fig.suptitle(
            f"UHRC XY Trajectory — {TC_ID}  {SCENARIO}   "
            f"[Trial {trial['trial_idx']}  ·  {out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig



def plot_distance(trial: dict, caption: bool = True):
    dist_log = trial["dist_log"].astype(float)
    metrics  = trial["metrics"]
    dyn_obs  = trial["dyn_obs"]
    t_ax     = np.arange(len(dist_log)) * DT

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    # Main distance curve
    ax.plot(t_ax, dist_log, color=ACCENT, lw=2.0, zorder=4,
            label="Distance to goal")
    ax.fill_between(t_ax, dist_log, GOAL_RADIUS,
                    where=dist_log > GOAL_RADIUS,
                    alpha=0.10, color=ACCENT, zorder=3)

    # Goal / near-miss reference lines
    ax.axhline(GOAL_RADIUS, color=GOAL_EDGE, ls="--", lw=1.4, zorder=5,
               label=f"Goal  ({GOAL_RADIUS} m)")
    ax.axhline(NEAR_MISS_TH, color="orange", ls=":", lw=1.2, zorder=5,
               label=f"Near-miss  ({NEAR_MISS_TH} m)")

    # Outcome marker
    ax.axvline(t_ax[-1], color=tcolor, ls=":", lw=1.3, alpha=0.7,
               label=outcome.capitalize())

    # Dynamic obstacle appearance markers
    for i, obs in enumerate(dyn_obs):
        t_app = obs["appear_step"] * DT
        if t_app <= t_ax[-1]:
            ax.axvline(t_app, color=DYN_ACTIVE, ls="--", lw=1.1,
                       alpha=0.75,
                       label=f"D{i} appears (s{obs['appear_step']})"
                             if i == 0 else f"D{i} (s{obs['appear_step']})")
            # Label at the distance value when obstacle appeared
            step_idx = min(obs["appear_step"], len(dist_log) - 1)
            ax.text(t_app + 0.05, float(dist_log[step_idx]),
                    f" D{i}", fontsize=ANNOT_FS - 1,
                    color=DYN_ACTIVE, va="bottom")

    # Reaction step marker
    if trial["reaction_step"] >= 0:
        rt = trial["reaction_step"] * DT
        ax.axvline(rt, color="red", ls="-.", lw=1.1, alpha=0.80,
                   label=f"Reaction  (s{trial['reaction_step']})")

    # Metric box
    ax.text(0.98, 0.97, _metric_box(metrics),
            transform=ax.transAxes,
            fontsize=METRIC_FS, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92))

    ax.set_xlabel("Time (s)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("Distance to goal (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG,
                    loc="upper right")
    leg.set_draggable(True)

    if caption:
        out = ("Success" if trial["success"]
               else "Collision" if trial["collision"] else "Timeout")
        fig.suptitle(
            f"UHRC Distance to Goal — {TC_ID}  {SCENARIO}   "
            f"[Trial {trial['trial_idx']}  ·  {out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig



def plot_control(trial: dict, caption: bool = True):
    ctrl_signals = [
        ("fz_log",        "$F_z$ (N)",            "#D97706"),
        ("tau_phi_log",   r"$\tau_\phi$ (N·m)",   "#7C3AED"),
        ("tau_theta_log", r"$\tau_\theta$ (N·m)", "#059669"),
        ("tau_psi_log",   r"$\tau_\psi$ (N·m)",   "#DC2626"),
    ]
    present = [(k, lbl, c) for k, lbl, c in ctrl_signals
               if trial.get(k) is not None and len(trial[k]) > 0]

    dyn_obs = trial["dyn_obs"]
    out     = ("Success" if trial["success"]
               else "Collision" if trial["collision"] else "Timeout")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    if present:
        for key, lbl, color in present:
            sig = trial[key]
            t_c = np.arange(len(sig)) * DT
            ax.plot(t_c, sig, color=color, lw=1.8, alpha=0.88,
                    label=lbl, zorder=3)
        ax.axhline(9.81, color=HOVER_REF, ls=":", lw=1.2,
                   label="$F_z$ hover  (9.81 N)", zorder=2)
        ax.axhline(0.0,  color=GRID_COL,  ls=":", lw=1.0,
                   label="Torque zero ref", zorder=2)

        # Mark dynamic appearance on control panel too
        for i, obs in enumerate(dyn_obs):
            t_app = obs["appear_step"] * DT
            t_max = len(trial[present[0][0]]) * DT
            if t_app <= t_max:
                ax.axvline(t_app, color=DYN_ACTIVE, ls="--",
                           lw=1.0, alpha=0.60,
                           label=f"D{i} appears" if i == 0 else "")
    else:
        ax.text(0.5, 0.5,
                "Control logs not saved in this .npz.\n"
                "Re-run the benchmark with --plot_trials to capture them.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=LABEL_FS, color=TEXT_SUB)

    ax.set_xlabel("Time (s)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("Control input", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG,
                    ncol=2, loc="upper right")
    leg.set_draggable(True)

    if caption:
        fig.suptitle(
            f"UHRC Control Inputs — {TC_ID}  {SCENARIO}   "
            f"[Trial {trial['trial_idx']}  ·  {out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig



def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a single dyn_random_pop trial .npz file "
                    "(3 separate windows)")
    p.add_argument("--trial",
                   default="benchmark_results/timeseries/TC04.npz",
                   help="Path to the .npz trial file")
    p.add_argument("--no_caption",  action="store_true",
                   help="Suppress figure titles")
    p.add_argument("--no_control",  action="store_true",
                   help="Skip the control-input window")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.trial):
        print(f"File not found: {args.trial}")
        return

    trial = load_trial(args.trial)

    m   = trial["metrics"]
    out = ("SUCCESS"   if trial["success"]
           else "COLLISION (dynamic)" if trial["dyn_hit"]
           else "COLLISION (static)"  if trial["collision"]
           else "TIMEOUT")
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  {TC_ID}  {SCENARIO}  |  Trial {trial['trial_idx']}")
    print(f"  File         : {args.trial}")
    print(f"  Outcome      : {out}")
    print(f"  Final dist   : {trial['final_dist']:.3f} m")
    print(f"  RMSE         : {m['rmse']:.3f} m")
    print(f"  IAE          : {m['iae']:.3f} m·s")
    print(f"  ITAE         : {m['itae']:.3f} m·s²")
    print(f"  PLR          : {m['plr']:.3f}")
    print(f"  Near-misses  : {trial['near_miss_count']} steps")
    rct = (f"step {trial['reaction_step']}"
           if trial["reaction_step"] >= 0 else "not detected")
    print(f"  Reaction     : {rct}")
    print(f"  Dyn obs      : {len(trial['dyn_obs'])}  "
          f"(appear steps: "
          f"{', '.join(str(o['appear_step']) for o in trial['dyn_obs'])})")
    print(f"{sep}\n")

    caption = not args.no_caption

    plot_environment(trial, caption=caption)
    plot_distance(trial,    caption=caption)
    if not args.no_control:
        plot_control(trial, caption=caption)

    plt.show()   


if __name__ == "__main__":
    main()