# view_test.py
# Visualization tool for UHRC benchmark trial .npz files.
#   TC-01  open_field          — static-only
#   TC-02  narrow_corridor     — static-only
#   TC-03  gap_navigation      — static-only
#   TC-04  dynamic_obstacles   — static + pop-in dynamic obstacles
#
#
# Usage:
#   python view_test.py                                             # default trial
#   python view_test.py --trial benchmark_results/trials/TC01.npz
#   python view_test.py --trial benchmark_results/trials/dynamic_obstacles_t000_s0.npz
#   python view_test.py --scenario narrow_corridor                  # first match in --dir
#   python view_test.py --scenario dynamic_obstacles --grid 2
#   python view_test.py --env_only
#   python view_test.py --no_control

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np


import matplotlib
matplotlib.use("TkAgg")          # swap to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap

#Change this to view trials (change /TC01.npz to view trials used in report )
TRIAL_PATH = "benchmark_results/trials/TC01.npz"
# Font sizes 
TITLE_FS  = 14
LABEL_FS  = 13
TICK_FS   = 11
LEGEND_FS = 11
ANNOT_FS  = 10
METRIC_FS = 10

# Palette 
BG         = "#FFFFFF"
PANEL_BG   = "#F7F8FA"
GRID_COL   = "#DADDE3"
OBS_FACE   = "#D6303080"
OBS_EDGE   = "#A31515"
DYN_EDGE   = "#7B1FA2"      # dynamic obstacle border
DYN_FACE   = "#CE93D880"    # dynamic obstacle fill (ghost before activation)
DYN_ACTIVE = "#9C27B0"      # dynamic obstacle trail / appearance markers
GOAL_FACE  = "#1E8B4420"
GOAL_EDGE  = "#1E8B44"
TEXT_MAIN  = "#1A1A2E"
TEXT_SUB   = "#555F70"
ACCENT     = "#2563EB"      # success trajectory
WARN       = "#D97706"      # timeout trajectory
DANGER     = "#DC2626"      # collision trajectory
HOVER_REF  = "#6B7280"

TRAJ_COLOR = {"success": ACCENT, "collision": DANGER, "timeout": WARN}

ARENA        = (-10.0, 10.0)
GOAL_RADIUS  = 0.5
NEAR_MISS_TH = 0.3
GRID_RES     = 0.1
DT           = 0.01

_TC_IDS = {
    "open_field":         "TC-01",
    "dense":              "TC-02",
    "narrow_corridor":    "TC-03",
    "gap_navigation":     "TC-04",
    "close_range":        "TC-05",
    "long_range":         "TC-06",
    "dynamic_obstacles":  "TC-10",
    "actuator_fault_sweep": "TC-11",
    "urban_corridor":     "TC-12",
    "urban_dense":        "TC-13",
}


#  GRID HELPERS

def _build_clearance_grid(circles, rects):
    lo, hi = ARENA
    xs = np.arange(lo, hi + GRID_RES, GRID_RES)
    ys = np.arange(lo, hi + GRID_RES, GRID_RES)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.full_like(xg, 20.0)
    for (cx, cy, _), r in circles:
        dist = np.minimum(dist, np.sqrt((xg - cx)**2 + (yg - cy)**2) - r)
    for cx, cy, hx, hy in rects:
        dx   = np.maximum(np.abs(xg - cx) - hx, 0.0)
        dy   = np.maximum(np.abs(yg - cy) - hy, 0.0)
        dist = np.minimum(dist, np.sqrt(dx**2 + dy**2))
    return np.clip(dist, 0.0, 5.0)


def _build_occupancy_grid(circles, rects):
    lo, hi = ARENA
    xs = np.arange(lo, hi + GRID_RES, GRID_RES)
    ys = np.arange(lo, hi + GRID_RES, GRID_RES)
    xg, yg = np.meshgrid(xs, ys)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for (cx, cy, _), r in circles:
        grid[np.sqrt((xg - cx)**2 + (yg - cy)**2) <= r] = 1.0
    for cx, cy, hx, hy in rects:
        grid[(xg >= cx-hx) & (xg <= cx+hx) &
             (yg >= cy-hy) & (yg <= cy+hy)] = 1.0
    return grid, [lo, hi, lo, hi]


def _draw_bg(ax, circles, rects):
    """Draw clearance heat-map + occupancy overlay on ax."""
    if circles or rects:
        clr      = _build_clearance_grid(circles, rects)
        cmap_clr = LinearSegmentedColormap.from_list(
            "clr", ["#C8D6E5", "#EEF2F7", PANEL_BG])
        ax.imshow(clr, extent=[-10, 10, -10, 10], origin="lower",
                  cmap=cmap_clr, alpha=0.70, vmin=0.0, vmax=5.0,
                  aspect="equal", zorder=0)
    occ, ext = _build_occupancy_grid(circles, rects)
    cmap_occ = LinearSegmentedColormap.from_list("occ", ["none", "#FFCDD2"])
    ax.imshow(occ, extent=ext, origin="lower", cmap=cmap_occ,
              alpha=0.55, vmin=0, vmax=1, aspect="equal", zorder=1)


def _draw_static_obstacles(ax, circles, rects):
    for (cx, cy, _), r in circles:
        ax.add_patch(Circle((cx, cy), r, lw=1.5,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))
    for cx, cy, hx, hy in rects:
        ax.add_patch(Rectangle((cx-hx, cy-hy), 2*hx, 2*hy, lw=1.5,
                                edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))


def _draw_dynamic_obstacles(ax, dyn_obs):
    """Render dynamic obstacles: ghost circle at original position + final position."""
    for i, obs in enumerate(dyn_obs):
        cx0, cy0, r = obs["cx0"], obs["cy0"], obs["r"]
        appear      = obs["appear_step"]
        trail       = obs["trail"]

        # Ghost at original position
        ax.add_patch(Circle((cx0, cy0), r, lw=1.8, ls="--",
                            edgecolor=DYN_EDGE, facecolor=DYN_FACE, zorder=3))
        ax.text(cx0, cy0, f"D{i}\n@s{appear}",
                ha="center", va="center",
                fontsize=ANNOT_FS - 2, color="#4A148C",
                fontweight="bold", zorder=5)

        # Trail from appearance step onwards (dynamic obstacles are static in this sim)
        if trail is not None and appear < len(trail):
            ax.plot(trail[appear:, 0], trail[appear:, 1],
                    color=DYN_ACTIVE, lw=1.2, ls=":", alpha=0.6, zorder=3)
            fx, fy = trail[-1]
            ax.add_patch(Circle((fx, fy), r, lw=1.5,
                                edgecolor=DYN_EDGE, facecolor="#E1BEE7",
                                alpha=0.55, zorder=3))


def _draw_goal(ax, goal):
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=True,
                        facecolor=GOAL_FACE, zorder=3))
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=False,
                        ls="--", lw=1.5, edgecolor=GOAL_EDGE, zorder=3))


def _draw_trajectory(ax, traj, tcolor):
    for i in range(len(traj) - 1):
        alpha = 0.20 + 0.80 * (i / max(len(traj) - 2, 1))
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                color=tcolor, lw=2.0, alpha=alpha,
                solid_capstyle="round", zorder=4)
    step = max(1, len(traj) // 15)
    for i in range(0, len(traj) - 2, step):
        dx  = traj[i+1, 0] - traj[i, 0]
        dy  = traj[i+1, 1] - traj[i, 1]
        mag = math.sqrt(dx**2 + dy**2) + 1e-9
        ax.annotate("",
                    xy=(traj[i, 0] + dx/mag*0.55, traj[i, 1] + dy/mag*0.55),
                    xytext=(traj[i, 0], traj[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=tcolor,
                                   lw=1.1, alpha=0.60),
                    zorder=5)


def _draw_markers(ax, start, goal, traj, tcolor):
    ax.plot(*start[:2], "o", ms=11, color="#16A34A",
            mec="white", mew=1.5, zorder=6)
    ax.plot(*goal[:2],  "*", ms=16, color="#F59E0B",
            mec="white", mew=1.0, zorder=6)
    ax.plot(*traj[-1, :2], "s", ms=10, color=tcolor,
            mec="white", mew=1.2, zorder=6)


def _ax_style(ax):
    ax.set_xlim(ARENA); ax.set_ylim(ARENA)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.set_xlabel("$x$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("$y$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)


def _metric_box(metrics: dict, include_itae: bool = False) -> str:
    s = (f"RMSE : {metrics['rmse']:.3f} m\n"
         f"IAE  : {metrics['iae']:.3f} m·s\n")
    if include_itae and "itae" in metrics:
        s += f"ITAE : {metrics['itae']:.3f} m·s²\n"
    s += f"PLR  : {metrics['plr']:.3f}"
    return s


#  LOAD

def _try(d, key, default=None):
    try:
        return d[key]
    except KeyError:
        return default


def _load_dyn_obs_list(d) -> list:
    """Parse dynamic_obs and dynamic_traj keys into a list of dicts."""
    dyn_arr  = _try(d, "dynamic_obs")
    dyn_traj = _try(d, "dynamic_traj")
    dyn_obs  = []
    if dyn_arr is not None and len(dyn_arr):
        for i in range(len(dyn_arr)):
            dyn_obs.append({
                "cx0":         float(dyn_arr[i, 0]),
                "cy0":         float(dyn_arr[i, 1]),
                "r":           float(dyn_arr[i, 2]),
                "appear_step": int(dyn_arr[i, 3]),
                "trail":       (dyn_traj[:, i, :] if dyn_traj is not None
                                and dyn_traj.ndim == 3 and dyn_traj.shape[1] > i
                                else None),
            })
    return dyn_obs


def load_trial(npz_path: str) -> dict:
    """
    Load a .npz saved 
      trajectory, dist_log, circles, rects, dynamic_obs, dynamic_traj,
      actions_post, fz_log, tau_*_log, success, collision, dynamic_obs_hit,
      near_miss_count, reaction_step, first_appear_step, rmse, iae, itae, plr
    """
    d = np.load(npz_path, allow_pickle=False)

    trajectory = d["trajectory"]      # [T+1, 3]
    dist_log   = d["dist_log"]        # [T]
    start      = d["start"]
    goal       = d["goal"]

    #Static obstacles 
    circ_arr = _try(d, "circles", np.zeros((0, 3), dtype=np.float32))
    circles  = [((float(circ_arr[i, 0]), float(circ_arr[i, 1]), 0.0),
                  float(circ_arr[i, 2]))
                for i in range(len(circ_arr))]

    rect_arr = _try(d, "rects", np.zeros((0, 4), dtype=np.float32))
    rects    = [(float(rect_arr[i, 0]), float(rect_arr[i, 1]),
                 float(rect_arr[i, 2]), float(rect_arr[i, 3]))
                for i in range(len(rect_arr))]

    # Dynamic obstacles 
    dyn_obs = _load_dyn_obs_list(d)

    # Control signals 
    fz_log        = _try(d, "fz_log")
    tau_phi_log   = _try(d, "tau_phi_log")
    tau_theta_log = _try(d, "tau_theta_log")
    tau_psi_log   = _try(d, "tau_psi_log")

    acts = _try(d, "actions_post")
    if acts is not None and acts.ndim == 2 and acts.shape[1] >= 4:
        if fz_log        is None: fz_log        = acts[:, 0]
        if tau_phi_log   is None: tau_phi_log   = acts[:, 1]
        if tau_theta_log is None: tau_theta_log = acts[:, 2]
        if tau_psi_log   is None: tau_psi_log   = acts[:, 3]

    # Outcome 
    final_dist = float(np.linalg.norm(trajectory[-1, :2] - goal[:2]))
    success    = bool(_try(d, "success",  np.array([final_dist < GOAL_RADIUS]))[0])
    collision  = bool(_try(d, "collision", np.array([False]))[0])
    dyn_hit    = bool(_try(d, "dynamic_obs_hit", np.array([False]))[0])
    timeout    = not success and not collision

    # ── Metrics — use pre-saved scalars, recompute if absent
    def _scalar(key, fallback):
        v = _try(d, key)
        return float(v[0]) if v is not None else fallback

    dist_arr = dist_log.astype(float)
    t_arr    = np.arange(len(dist_arr)) * DT
    rmse = _scalar("rmse", float(np.sqrt(np.mean(dist_arr**2))))
    iae  = _scalar("iae",  float(trapezoid(np.abs(dist_arr), t_arr)))
    itae = _scalar("itae", float(trapezoid(t_arr * np.abs(dist_arr), t_arr)))
    plr_fallback = (float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)))
                    / max(float(np.linalg.norm(goal[:2] - start[:2])), 1e-6))
    plr = _scalar("plr", plr_fallback)

    near_miss_count   = int(_try(d, "near_miss_count",   np.array([0]))[0])
    reaction_step     = int(_try(d, "reaction_step",     np.array([-1]))[0])
    first_appear_step = int(_try(d, "first_appear_step", np.array([-1]))[0])

    # Filename parsing  scenario 
    stem  = Path(npz_path).stem
    if "_t" in stem and "_s" in stem:
        base  = stem.rsplit("_s", 1)[0]
        parts = base.rsplit("_t", 1)
        scenario  = parts[0] if len(parts) == 2 else "unknown"
        try:    trial_idx = int(parts[1])
        except: trial_idx = 0
    elif "_trial" in stem:
        parts = stem.rsplit("_trial", 1)
        scenario  = parts[0] if len(parts) == 2 else "unknown"
        try:    trial_idx = int(parts[1])
        except: trial_idx = 0
    else:
        scenario  = stem
        trial_idx = 0

    return {
        "trajectory":        trajectory,
        "start":             start,
        "goal":              goal,
        "dist_log":          dist_log,
        "circles":           circles,
        "rects":             rects,
        "dyn_obs":           dyn_obs,        # list of dicts; empty for non-TC-10
        "fz_log":            fz_log,
        "tau_phi_log":       tau_phi_log,
        "tau_theta_log":     tau_theta_log,
        "tau_psi_log":       tau_psi_log,
        "success":           success,
        "collision":         collision,
        "dyn_hit":           dyn_hit,
        "timeout":           timeout,
        "final_dist":        final_dist,
        "near_miss_count":   near_miss_count,
        "reaction_step":     reaction_step,
        "first_appear_step": first_appear_step,
        "scenario":          scenario,
        "trial_idx":         trial_idx,
        "metrics": {
            "rmse": rmse,
            "iae":  iae,
            "itae": itae,
            "plr":  plr,
        },
    }



#  Plot 1 — XY TRAJECTORY

def plot_environment(trial: dict, caption: bool = True):
    circles  = trial["circles"]
    rects    = trial["rects"]
    dyn_obs  = trial["dyn_obs"]
    traj     = trial["trajectory"]
    start    = trial["start"]
    goal     = trial["goal"]
    metrics  = trial["metrics"]
    sc       = trial["scenario"]
    tc       = _TC_IDS.get(sc, "??")
    is_dyn   = len(dyn_obs) > 0

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    _draw_bg(ax, circles, rects)
    _draw_static_obstacles(ax, circles, rects)
    if is_dyn:
        _draw_dynamic_obstacles(ax, dyn_obs)
    _draw_goal(ax, goal)
    if len(traj) > 1:
        _draw_trajectory(ax, traj, tcolor)
    _draw_markers(ax, start, goal, traj, tcolor)

    # Coordinate annotation 
    def fmt(v):
        return (f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
                if len(v) >= 3 else f"({v[0]:.2f}, {v[1]:.2f})")

    coord_text = (f"Start :  {fmt(start)}\n"
                  f"Goal  :  {fmt(goal)}\n"
                  f"Final :  {fmt(traj[-1])}")
    ax.text(0.02, 0.02, coord_text, transform=ax.transAxes,
            fontsize=ANNOT_FS, color=TEXT_MAIN, family="monospace",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92), zorder=8)

    # Metrics box 
    ax.text(0.98, 0.98, _metric_box(metrics, include_itae=is_dyn),
            transform=ax.transAxes,
            fontsize=METRIC_FS, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92), zorder=8)

    # dynamic info box 
    if is_dyn:
        appear_steps = ", ".join(str(o["appear_step"]) for o in dyn_obs)
        rct = (f"step {trial['reaction_step']}"
               if trial["reaction_step"] >= 0 else "not detected")
        dyn_text = (f"Dynamic obs   : {len(dyn_obs)}\n"
                    f"Appear steps  : {appear_steps}\n"
                    f"Near-misses   : {trial['near_miss_count']} steps\n"
                    f"Reaction      : {rct}")
        ax.text(0.02, 0.98, dyn_text, transform=ax.transAxes,
                fontsize=ANNOT_FS - 1, color=TEXT_MAIN, family="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                          edgecolor=DYN_EDGE, alpha=0.88), zorder=8)

    _ax_style(ax)

    # Legend 
    col_label = ("Dynamic obstacle hit" if trial["dyn_hit"]
                 else outcome.capitalize())
    handles = [
        mpatches.Patch(color=tcolor, alpha=0.85,
                       label=f"{col_label}  (d = {trial['final_dist']:.2f} m)"),
        mpatches.Patch(color=OBS_EDGE, alpha=0.55, label="Static obstacle"),
    ]
    if is_dyn:
        handles.append(mpatches.Patch(color=DYN_EDGE, alpha=0.55,
                                      label="Dynamic obstacle (pop-in)"))
    handles += [
        mpatches.Patch(color=GOAL_EDGE, alpha=0.55,
                       label=f"Goal zone  (r = {GOAL_RADIUS} m)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#16A34A", ms=10, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#F59E0B", ms=12, label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=tcolor, ms=9, label="Final position"),
    ]
    loc = "lower right" if is_dyn else "upper left"
    leg = ax.legend(handles=handles, loc=loc,
                    fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    if caption:
        out = ("Success" if trial["success"]
               else "Collision (dynamic)" if trial["dyn_hit"]
               else "Collision (static)"  if trial["collision"] else "Timeout")
        fig.suptitle(
            f"UHRC XY Trajectory  [{tc}  ·  {sc.replace('_', ' ').title()}"
            f"  ·  Trial {trial['trial_idx']}  ·  {out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


#  Plot 2 — DISTANCE TO GOAL

def plot_distance(trial: dict, caption: bool = True):
    dist_log = trial["dist_log"].astype(float)
    metrics  = trial["metrics"]
    dyn_obs  = trial["dyn_obs"]
    t_ax     = np.arange(len(dist_log)) * DT
    sc       = trial["scenario"]
    tc       = _TC_IDS.get(sc, "??")
    is_dyn   = len(dyn_obs) > 0

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(t_ax, dist_log, color=ACCENT, lw=2.0, zorder=4,
            label="Distance to goal")
    ax.fill_between(t_ax, dist_log, GOAL_RADIUS,
                    where=dist_log > GOAL_RADIUS,
                    alpha=0.10, color=ACCENT, zorder=3)
    ax.axhline(GOAL_RADIUS, color=GOAL_EDGE, ls="--", lw=1.4, zorder=5,
               label=f"Goal  ({GOAL_RADIUS} m)")

    if is_dyn:
        ax.axhline(NEAR_MISS_TH, color="orange", ls=":", lw=1.2, zorder=5,
                   label=f"Near-miss  ({NEAR_MISS_TH} m)")
        for i, obs in enumerate(dyn_obs):
            t_app = obs["appear_step"] * DT
            if t_app <= t_ax[-1]:
                lbl = (f"D{i} appears (s{obs['appear_step']})"
                       if i == 0 else f"D{i} (s{obs['appear_step']})")
                ax.axvline(t_app, color=DYN_ACTIVE, ls="--",
                           lw=1.1, alpha=0.75, label=lbl)
                step_idx = min(obs["appear_step"], len(dist_log) - 1)
                ax.text(t_app + 0.05, float(dist_log[step_idx]),
                        f" D{i}", fontsize=ANNOT_FS - 1,
                        color=DYN_ACTIVE, va="bottom")
        if trial["reaction_step"] >= 0:
            rt = trial["reaction_step"] * DT
            ax.axvline(rt, color="red", ls="-.", lw=1.1, alpha=0.80,
                       label=f"Reaction  (s{trial['reaction_step']})")

    ax.axvline(t_ax[-1], color=tcolor, ls=":", lw=1.3, alpha=0.70,
               label=outcome.capitalize())

    ax.text(0.98, 0.97, _metric_box(metrics, include_itae=is_dyn),
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
            f"UHRC Distance to Goal — {tc}  ·  "
            f"{sc.replace('_', ' ').title()}   [{out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


#  Plot 3 — CONTROL INPUTS

def plot_control(trial: dict, caption: bool = True):
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")
    dyn_obs = trial["dyn_obs"]
    is_dyn  = len(dyn_obs) > 0
    out     = ("Success" if trial["success"]
               else "Collision" if trial["collision"] else "Timeout")

    ctrl_signals = [
        ("fz_log",        "$F_z$ (N)",            "#D97706"),
        ("tau_phi_log",   r"$\tau_\phi$ (N·m)",   "#7C3AED"),
        ("tau_theta_log", r"$\tau_\theta$ (N·m)", "#059669"),
        ("tau_psi_log",   r"$\tau_\psi$ (N·m)",   "#DC2626"),
    ]
    present = [(k, lbl, c) for k, lbl, c in ctrl_signals
               if trial.get(k) is not None and len(trial[k]) > 0]

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
        if is_dyn:
            for i, obs in enumerate(dyn_obs):
                t_app = obs["appear_step"] * DT
                t_max = len(trial[present[0][0]]) * DT
                if t_app <= t_max:
                    ax.axvline(t_app, color=DYN_ACTIVE, ls="--",
                               lw=1.0, alpha=0.60,
                               label=f"D{i} appears" if i == 0 else "")
    else:
        ax.text(0.5, 0.5,
                "Control logs not saved in this .npz\n"
                "(re-run benchmark with --save_timeseries)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=LABEL_FS, color=TEXT_SUB)

    ax.set_xlabel("Time (s)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("Control inputs", fontsize=LABEL_FS, color=TEXT_SUB)
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
            f"UHRC Control Inputs — {tc}  ·  "
            f"{sc.replace('_', ' ').title()}   [{out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig



def plot_env_only(trial: dict, caption: bool = True):
    circles = trial["circles"]
    rects   = trial["rects"]
    dyn_obs = trial["dyn_obs"]
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")
    is_dyn  = len(dyn_obs) > 0

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    _draw_bg(ax, circles, rects)
    _draw_static_obstacles(ax, circles, rects)
    if is_dyn:
        _draw_dynamic_obstacles(ax, dyn_obs)
    _draw_goal(ax, trial["goal"])
    ax.plot(*trial["start"][:2], "o", ms=11, color="#16A34A",
            mec="white", mew=1.5, zorder=5, label="Start")
    ax.plot(*trial["goal"][:2],  "*", ms=15, color="#F59E0B",
            mec="white", mew=1.0, zorder=5, label="Goal")

    _ax_style(ax)
    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.90,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    if caption:
        fig.suptitle(f"Test Scenario — {tc}  ·  {sc.replace('_', ' ').title()}",
                     fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig



def _draw_env_on_ax(ax, trial, label_prefix=""):
    circles = trial["circles"]
    rects   = trial["rects"]
    dyn_obs = trial["dyn_obs"]
    traj    = trial["trajectory"]
    start   = trial["start"]
    goal    = trial["goal"]
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")
    metrics = trial["metrics"]
    is_dyn  = len(dyn_obs) > 0

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    ax.set_facecolor(PANEL_BG)
    _draw_bg(ax, circles, rects)
    _draw_static_obstacles(ax, circles, rects)
    if is_dyn:
        _draw_dynamic_obstacles(ax, dyn_obs)
    _draw_goal(ax, goal)
    if len(traj) > 1:
        _draw_trajectory(ax, traj, tcolor)
    _draw_markers(ax, start, goal, traj, tcolor)

    ax.text(0.98, 0.98, _metric_box(metrics, include_itae=is_dyn),
            transform=ax.transAxes,
            fontsize=METRIC_FS - 1, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.40", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92), zorder=8)

    out_lbl = ("Success"   if trial["success"]
               else "Dyn hit"   if trial["dyn_hit"]
               else "Collision" if trial["collision"] else "Timeout")
    handles = [
        mpatches.Patch(color=tcolor, alpha=0.85,
                       label=f"{out_lbl}  (d = {trial['final_dist']:.2f} m)"),
        mpatches.Patch(color=OBS_EDGE, alpha=0.55, label="Static obs"),
    ]
    if is_dyn:
        handles.append(mpatches.Patch(color=DYN_EDGE, alpha=0.55,
                                      label="Dynamic obs"))
    handles += [
        mpatches.Patch(color=GOAL_EDGE, alpha=0.55,
                       label=f"Goal  (r = {GOAL_RADIUS} m)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#16A34A", ms=9, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#F59E0B", ms=11, label="Goal"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    fontsize=LEGEND_FS - 1, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    ax.set_title(f"{label_prefix}{tc}  ·  {sc.replace('_', ' ').title()}",
                 fontsize=TITLE_FS - 1, fontweight="bold",
                 color=TEXT_MAIN, pad=5)
    _ax_style(ax)


def plot_grid(trials: list, cols: int = 2):
    fig = plt.figure(figsize=(7 * cols, 7), facecolor=BG)
    gs  = mgridspec.GridSpec(1, cols, figure=fig,
                             left=0.05, right=0.97,
                             top=0.88, bottom=0.06, wspace=0.28)
    for col, trial in enumerate(trials[:cols]):
        ax = fig.add_subplot(gs[0, col])
        _draw_env_on_ax(ax, trial, label_prefix=f"({chr(97 + col)}) ")

    scenarios = "  ·  ".join(
        f"{_TC_IDS.get(t['scenario'], '??')} "
        f"{t['scenario'].replace('_', ' ').title()}"
        for t in trials[:cols])
    fig.suptitle(f"UHRC Navigation Trials — {scenarios}",
                 fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)
    fig.tight_layout()
    return fig



def parse_args():
    p = argparse.ArgumentParser(
        description="UHRC benchmark trial visualizer (all scenarios including TC-10)")
    p.add_argument("--trial",      type=str,
                default=TRIAL_PATH,
                   help="Path to a single .npz trial file")
    p.add_argument("--scenario",   type=str, default=None,
                   help="Scenario name filter for --dir search (use with --grid)")
    p.add_argument("--dir",        type=str,
                   default="benchmark_results/trials",
                   help="Directory to search for .npz files")
    p.add_argument("--grid",       type=str, default=None,
                   help="Side-by-side grid of N trials, e.g. --grid 2")
    p.add_argument("--env_only",   action="store_true",
                   help="Environment-map-only window (for Methodology section)")
    p.add_argument("--no_caption", action="store_true",
                   help="Suppress figure titles")
    p.add_argument("--no_control", action="store_true",
                   help="Skip the control-input window")
    return p.parse_args()


def main():
    args  = parse_args()
    n_col = int(args.grid) if args.grid else 1

    if args.scenario:
        patterns = [
            os.path.join(args.dir, f"{args.scenario}_t*_s*.npz"),
            os.path.join(args.dir, f"{args.scenario}_trial*.npz"),
        ]
        paths = []
        for pat in patterns:
            paths.extend(sorted(glob.glob(pat)))
        paths = paths[:max(n_col, 1)]
    elif args.grid:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.npz")))[:n_col]
    else:
        paths = [args.trial]

    if not paths:
        print("No trial files found.")
        return

    trials = [load_trial(p) for p in paths]

    for trial in trials:
        m   = trial["metrics"]
        sc  = trial["scenario"]
        tc  = _TC_IDS.get(sc, "??")
        is_dyn = len(trial["dyn_obs"]) > 0
        out = ("SUCCESS"   if trial["success"]
               else "COLLISION (dynamic)" if trial["dyn_hit"]
               else "COLLISION (static)"  if trial["collision"]
               else "TIMEOUT")
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  {tc}  |  {sc}  |  Trial {trial['trial_idx']}")
        print(f"  File         : {paths[trials.index(trial)]}")
        print(f"  Outcome      : {out}")
        print(f"  Final dist   : {trial['final_dist']:.3f} m")
        print(f"  RMSE         : {m['rmse']:.3f} m")
        print(f"  IAE          : {m['iae']:.3f} m·s")
        print(f"  ITAE         : {m['itae']:.3f} m·s²")
        print(f"  PLR          : {m['plr']:.3f}")
        if is_dyn:
            print(f"  Dyn obs      : {len(trial['dyn_obs'])}")
            print(f"  Near-misses  : {trial['near_miss_count']} steps")
            rct = (f"step {trial['reaction_step']}"
                   if trial["reaction_step"] >= 0 else "not detected")
            print(f"  Reaction     : {rct}")
        print(sep)

    caption = not args.no_caption

    if args.env_only:
        plot_env_only(trials[0], caption=caption)
    elif args.grid and len(trials) > 1:
        plot_grid(trials, cols=n_col)
    else:
        trial = trials[0]
        plot_environment(trial, caption=caption)
        plot_distance(trial,    caption=caption)
        if not args.no_control:
            plot_control(trial, caption=caption)

    plt.show()


if __name__ == "__main__":
    main()