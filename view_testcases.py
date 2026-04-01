# view_test.py
#This script provides visualization tools for UHRC benchmark trial presented in the report figures
#In the benchmark_results/trials/ directory, you can find .npz files containing trial data for each test scenario
#presented named TC01.npz, TC02.npz, etc.

# Usage:
#   python view_test.py
#   python view_test.py --trial benchmark_results/trials/TC01.npz
#   python view_test.py --scenario urban_mixed --grid 2
#   python view_test.py --env_only



import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors  import LinearSegmentedColormap
from pathlib import Path

#font sizes for readability
TITLE_FS   = 14
LABEL_FS   = 13
TICK_FS    = 11
LEGEND_FS  = 11
ANNOT_FS   = 10
METRIC_FS  = 10

#Color codes
BG        = "#FFFFFF"
PANEL_BG  = "#F7F8FA"
GRID_COL  = "#DADDE3"
OBS_FACE  = "#D6303080"
OBS_EDGE  = "#A31515"
GOAL_FACE = "#1E8B4420"
GOAL_EDGE = "#1E8B44"
TEXT_MAIN = "#1A1A2E"
TEXT_SUB  = "#555F70"
ACCENT    = "#2563EB"       # trajectory (success)
WARN      = "#D97706"       # timeout
DANGER    = "#DC2626"       # collision
HOVER_REF = "#6B7280"

ARENA       = (-10.0, 10.0)
GOAL_RADIUS = 0.5
GRID_RES    = 0.1
DT          = 0.01          # simulation timestep (seconds)

_TC_IDS = {
    "open_field":      "TC-01", "sparse":          "TC-02",
    "moderate":        "TC-03", "dense":            "TC-04",
    "narrow_corridor": "TC-02", "gap_navigation":   "TC-03",
    "cluttered":       "TC-07", "close_range":      "TC-08",
    "long_range":      "TC-09", "urban_mixed":      "TC-10",
    "urban_dense":     "TC-11", "urban_corridor":   "TC-12",
}

TRAJ_COLOR = {"success": ACCENT, "collision": DANGER, "timeout": WARN}



def compute_metrics(trial):
    """
    Returns a dict with RMSE, IAE, and Path Length Ratio (PLR).

    RMSE  = sqrt( mean( dist_log² ) )   — root-mean-square distance to goal
    IAE   = sum( dist_log ) * DT         — integral absolute error (m·s)
    PLR   = path_length / straight_line_dist — 1.0 = perfect straight line
    """
    dist_log = trial["dist_log"].astype(float)
    traj     = trial["trajectory"]
    start    = trial["start"]
    goal     = trial["goal"]

    # RMSE
    rmse = float(np.sqrt(np.mean(dist_log ** 2)))

    # IAE (trapezoidal integration)
    t_arr = np.arange(len(dist_log)) * DT
    iae   = float(np.trapezoid(np.abs(dist_log), t_arr))

    # Path length (sum of Euclidean step distances)
    diffs       = np.diff(traj[:, :2], axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Straight-line distance start → goal
    straight = float(np.linalg.norm(goal[:2] - start[:2]))
    plr      = path_length / max(straight, 1e-6)

    return {
        "rmse":        rmse,
        "iae":         iae,
        "path_length": path_length,
        "straight":    straight,
        "plr":         plr,
    }



def build_clearance_grid(circles, rects, res=GRID_RES):
    lo, hi = ARENA
    xs = np.arange(lo, hi + res, res)
    ys = np.arange(lo, hi + res, res)
    xg, yg = np.meshgrid(xs, ys)
    dist = np.full_like(xg, 20.0)
    for (cx, cy, _), r in circles:
        dist = np.minimum(dist, np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2) - r)
    for cx, cy, hx, hy in rects:
        dx   = np.maximum(np.abs(xg - cx) - hx, 0.0)
        dy   = np.maximum(np.abs(yg - cy) - hy, 0.0)
        dist = np.minimum(dist, np.sqrt(dx ** 2 + dy ** 2))
    return np.clip(dist, 0.0, 5.0)


def build_occupancy_grid(circles, rects, res=GRID_RES):
    lo, hi = ARENA
    xs = np.arange(lo, hi + res, res)
    ys = np.arange(lo, hi + res, res)
    xg, yg = np.meshgrid(xs, ys)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for (cx, cy, _), r in circles:
        grid[np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2) <= r] = 1.0
    for cx, cy, hx, hy in rects:
        grid[
            (xg >= cx - hx) & (xg <= cx + hx) &
            (yg >= cy - hy) & (yg <= cy + hy)
        ] = 1.0
    return grid, [lo, hi, lo, hi]


def _try_load(d, key):
    try:
        return d[key]
    except KeyError:
        return None



def load_trial(npz_path):
    d        = np.load(npz_path, allow_pickle=False)
    circ_arr = d["circles"]
    rect_arr = d["rects"]

    trial = {
        "trajectory":    d["trajectory"],
        "start":         d["start"],
        "goal":          d["goal"],
        "dist_log":      d["dist_log"],
        "fz_log":        _try_load(d, "fz_log"),
        "tau_phi_log":   _try_load(d, "tau_phi_log"),
        "tau_theta_log": _try_load(d, "tau_theta_log"),
        "tau_psi_log":   _try_load(d, "tau_psi_log"),
        "circles": [
            ((float(circ_arr[i, 0]), float(circ_arr[i, 1]), 0.0),
              float(circ_arr[i, 2]))
            for i in range(len(circ_arr))
        ],
        "rects": [
            (float(rect_arr[i, 0]), float(rect_arr[i, 1]),
             float(rect_arr[i, 2]), float(rect_arr[i, 3]))
            for i in range(len(rect_arr))
        ],
    }

    final            = np.linalg.norm(
        trial["trajectory"][-1, :2] - trial["goal"][:2])
    T                = len(trial["trajectory"])
    trial["final_dist"]  = float(final)
    trial["success"]     = final < GOAL_RADIUS
    trial["collision"]   = not trial["success"] and T < 50
    trial["timeout"]     = not trial["success"] and not trial["collision"]

    fname            = Path(npz_path).stem
    parts            = fname.rsplit("_trial", 1)
    trial["scenario"]    = parts[0] if len(parts) == 2 else "unknown"
    trial["trial_idx"]   = int(parts[1]) if len(parts) == 2 else 0
    trial["npz_path"]    = str(npz_path)

    # Pre-compute metrics once
    trial["metrics"] = compute_metrics(trial)
    return trial


# Environment with XY trajectory rep occupancy grid

def plot_environment(trial, caption=True):
    """
    2D top-down view: obstacles, trajectory, start/goal, metric box.
    Opens in its own figure window.
    """
    circles = trial["circles"]
    rects   = trial["rects"]
    traj    = trial["trajectory"]
    start   = trial["start"]
    goal    = trial["goal"]
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")
    metrics = trial["metrics"]

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    # Clearance heatmap
    if circles or rects:
        clr      = build_clearance_grid(circles, rects)
        cmap_clr = LinearSegmentedColormap.from_list(
            "clr_light", ["#C8D6E5", "#EEF2F7", PANEL_BG])
        ax.imshow(clr, extent=[-10, 10, -10, 10], origin="lower",
                  cmap=cmap_clr, alpha=0.70, vmin=0.0, vmax=5.0,
                  aspect="equal", zorder=0)

    # Occupancy overlay
    occ, ext = build_occupancy_grid(circles, rects)
    cmap_occ = LinearSegmentedColormap.from_list("occ_light", ["none", "#FFCDD2"])
    ax.imshow(occ, extent=ext, origin="lower", cmap=cmap_occ,
              alpha=0.55, vmin=0, vmax=1, aspect="equal", zorder=1)

    # Obstacle outlines
    for (cx, cy, _), r in circles:
        ax.add_patch(Circle((cx, cy), r, lw=1.5,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))
    for cx, cy, hx, hy in rects:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, lw=1.5,
                                edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))

    # Goal zone
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=True,
                        facecolor=GOAL_FACE, zorder=3))
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=False,
                        ls="--", lw=1.5, edgecolor=GOAL_EDGE, zorder=3))

    # Trajectory with gradient opacity
    if len(traj) > 1:
        for i in range(len(traj) - 1):
            alpha = 0.20 + 0.80 * (i / max(len(traj) - 2, 1))
            ax.plot(traj[i:i + 2, 0], traj[i:i + 2, 1],
                    color=tcolor, lw=2.0, alpha=alpha,
                    solid_capstyle="round", zorder=4)

    # Direction arrows every ~15 steps
    step = max(1, len(traj) // 15)
    for i in range(0, len(traj) - 2, step):
        dx  = traj[i + 1, 0] - traj[i, 0]
        dy  = traj[i + 1, 1] - traj[i, 1]
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
        ax.annotate("",
                    xy=(traj[i, 0] + dx / mag * 0.55,
                        traj[i, 1] + dy / mag * 0.55),
                    xytext=(traj[i, 0], traj[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=tcolor,
                                   lw=1.1, alpha=0.60),
                    zorder=5)

    # Markers
    ax.plot(*start[:2], "o", ms=11, color="#16A34A",
            mec="white", mew=1.5, zorder=6)
    ax.plot(*goal[:2],  "*", ms=16, color="#F59E0B",
            mec="white", mew=1.0, zorder=6)
    ax.plot(*traj[-1, :2], "s", ms=10, color=tcolor,
            mec="white", mew=1.2, zorder=6)

    # ── Coordinate annotation ─────────────────────────────────────────────
    def fmt3(v):
        return (f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
                if len(v) >= 3 else f"({v[0]:.2f}, {v[1]:.2f})")

    coord_text = (
        f"Start :  {fmt3(start)}\n"
        f"Goal  :  {fmt3(goal)}\n"
        f"Final :  {fmt3(traj[-1])}"
    )
    ax.text(0.02, 0.02, coord_text,
            transform=ax.transAxes,
            fontsize=ANNOT_FS, color=TEXT_MAIN, family="monospace",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92),
            zorder=8)

    # Metric annotation
    metric_text = (
        f"RMSE : {metrics['rmse']:.3f} m\n"
        f"IAE  : {metrics['iae']:.3f} m·s\n"
        f"PLR  : {metrics['plr']:.3f}"
    )
    ax.text(0.98, 0.98, metric_text,
            transform=ax.transAxes,
            fontsize=METRIC_FS, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92),
            zorder=8)

    # Axis cosmetics
    ax.set_xlabel("$x$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("$y$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_xlim(ARENA)
    ax.set_ylim(ARENA)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    # Legend
    handles = [
        mpatches.Patch(color=tcolor, alpha=0.85,
                       label=f"{outcome.capitalize()}  "
                             f"(final dist = {trial['final_dist']:.2f} m)"),
        mpatches.Patch(color=OBS_EDGE, alpha=0.55, label="Obstacle"),
        mpatches.Patch(color=GOAL_EDGE, alpha=0.55,
                       label=f"Goal zone  (r = {GOAL_RADIUS} m)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#16A34A", ms=10, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#F59E0B", ms=12, label="Goal"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=tcolor, ms=9, label="Final position"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    if caption:
        tc_str = f"{tc}  ·  {sc.replace('_', ' ').title()}"
        out    = ("Success" if trial["success"]
                  else "Collision" if trial["collision"] else "Timeout")
        fig.suptitle(
            f"UHRC XY Trajectory — TC03 Gap Navigation   "
            f"[Trial {trial['trial_idx']}  ·  {out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


# Distance-to-goal over time  

def plot_distance(trial, caption=True):
    """
    Distance-to-goal time series with RMSE / IAE annotations.
    Opens in its own figure window.
    """
    dist_log = trial["dist_log"]
    metrics  = trial["metrics"]
    t_ax     = np.arange(len(dist_log)) * DT
    sc       = trial["scenario"]
    tc       = _TC_IDS.get(sc, "??")
    outcome  = ("success"   if trial["success"]
                else "collision" if trial["collision"] else "timeout")
    tcolor   = TRAJ_COLOR[outcome]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(t_ax, dist_log, color=ACCENT, lw=2.0, zorder=3,
            label="Distance to goal")
    ax.fill_between(t_ax, dist_log, GOAL_RADIUS,
                    where=dist_log > GOAL_RADIUS,
                    alpha=0.12, color=ACCENT, zorder=2)
    ax.axhline(GOAL_RADIUS, color=GOAL_EDGE, ls="--", lw=1.4, zorder=4,
               label=f"Goal threshold  ({GOAL_RADIUS} m)")

    # Outcome marker at final step
    ax.axvline(t_ax[-1], color=tcolor, ls=":", lw=1.3, alpha=0.7,
               label=f"{outcome.capitalize()}")

    # Metric annotation box
    metric_text = (
        f"RMSE : {metrics['rmse']:.3f} m\n"
        f"IAE  : {metrics['iae']:.3f} m·s\n"
        f"PLR  : {metrics['plr']:.3f}"
    )
    ax.text(0.98, 0.97, metric_text,
            transform=ax.transAxes,
            fontsize=METRIC_FS, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92))

    ax.set_xlabel("Time (s)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("Distance to goal (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    if caption:
        out = ("Success" if trial["success"]
               else "Collision" if trial["collision"] else "Timeout")
        fig.suptitle(
            f"UHRC Distance to Goal — {tc}  ·  "
            f"{sc.replace('_', ' ').title()}   [{out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


#  Control inputs U1–U4 plotted

def plot_control(trial, caption=True):
    """
    Control input time series: Fz (N), τ_φ, τ_θ, τ_ψ (N·m).
    Opens in its own figure window.
    """
    sc  = trial["scenario"]
    tc  = _TC_IDS.get(sc, "??")
    out = ("Success" if trial["success"]
           else "Collision" if trial["collision"] else "Timeout")

    ctrl_signals = [
        ("fz_log",        "$F_z$ (N)",            "#D97706"),
        ("tau_phi_log",   r"$\tau_\phi$ (N·m)",   "#7C3AED"),
        ("tau_theta_log", r"$\tau_\theta$ (N·m)", "#059669"),
        ("tau_psi_log",   r"$\tau_\psi$ (N·m)",   "#DC2626"),
    ]

    # Count how many signals are actually present
    present = [(k, lbl, c) for k, lbl, c in ctrl_signals
               if trial.get(k) is not None and len(trial[k]) > 0]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    if present:
        for key, ylabel_lbl, color in present:
            sig = trial[key]
            t_c = np.arange(len(sig)) * DT
            ax.plot(t_c, sig, color=color, lw=1.8, alpha=0.88,
                    label=ylabel_lbl, zorder=3)

        ax.axhline(9.81, color=HOVER_REF, ls=":", lw=1.2,
                   label="$F_z$ hover ref  (9.81 N)", zorder=2)
        ax.axhline(0.0,  color=GRID_COL,  ls=":", lw=1.0,
                   label="Torque zero ref", zorder=2)
    else:
        ax.text(0.5, 0.5,
                "Control logs not saved in .npz\n"
                "Add  fz_log / tau_phi_log / tau_theta_log / tau_psi_log\n"
                "to your benchmark trial save call.",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=LABEL_FS, color=TEXT_SUB)

    ax.set_xlabel("Time (s)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("Control inputs", fontsize=LABEL_FS, color=TEXT_SUB)
    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG,
                    ncol=2, loc="upper right")
    leg.set_draggable(True)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    if caption:
        fig.suptitle(
            f"UHRC Control Inputs — {tc}  ·  "
            f"{sc.replace('_', ' ').title()}   [{out}]",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


# Drones Environment  plot 

def plot_env_only(trial, caption=True):
    """
    Just the 2D environment map — no trajectory, no time series.
    Useful for the Methodology section describing test scenarios.
    """
    circles = trial["circles"]
    rects   = trial["rects"]
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    if circles or rects:
        clr  = build_clearance_grid(circles, rects)
        cmap = LinearSegmentedColormap.from_list(
            "clr_light", ["#C8D6E5", "#EEF2F7", PANEL_BG])
        ax.imshow(clr, extent=[-10, 10, -10, 10], origin="lower",
                  cmap=cmap, alpha=0.65, vmin=0, vmax=5,
                  aspect="equal", zorder=0)

    occ, ext = build_occupancy_grid(circles, rects)
    cmap_occ = LinearSegmentedColormap.from_list("occ", ["none", "#FFCDD2"])
    ax.imshow(occ, extent=ext, origin="lower", cmap=cmap_occ,
              alpha=0.5, vmin=0, vmax=1, aspect="equal", zorder=1)

    for (cx, cy, _), r in circles:
        ax.add_patch(Circle((cx, cy), r, lw=1.5,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))
    for cx, cy, hx, hy in rects:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, lw=1.5,
                                edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))

    ax.add_patch(Circle(trial["goal"][:2], GOAL_RADIUS, fill=True,
                        facecolor=GOAL_FACE, zorder=3))
    ax.add_patch(Circle(trial["goal"][:2], GOAL_RADIUS, fill=False,
                        ls="--", lw=1.5, edgecolor=GOAL_EDGE, zorder=3))
    ax.plot(*trial["start"][:2], "o", ms=11, color="#16A34A",
            mec="white", mew=1.5, zorder=5, label="Start")
    ax.plot(*trial["goal"][:2],  "*", ms=15, color="#F59E0B",
            mec="white", mew=1.0, zorder=5, label="Goal")

    ax.set_xlabel("$x$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("$y$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_xlim(ARENA)
    ax.set_ylim(ARENA)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    leg = ax.legend(fontsize=LEGEND_FS, framealpha=0.90,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)

    if caption:
        fig.suptitle(
            f"Test Scenario — {tc}  ·  {sc.replace('_', ' ').title()}",
            fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)

    fig.tight_layout()
    return fig


# Grid comparison plot for easy comparing

def plot_grid(trials, cols=2):

    import matplotlib.gridspec as mgridspec

    fig = plt.figure(figsize=(7 * cols, 7), facecolor=BG)
    gs  = mgridspec.GridSpec(1, cols, figure=fig,
                             left=0.05, right=0.97,
                             top=0.88, bottom=0.06,
                             wspace=0.28)

    for col, trial in enumerate(trials[:cols]):
        ax = fig.add_subplot(gs[0, col])
        draw_env_on_ax(ax, trial, label_prefix=f"({chr(97 + col)}) ")

    scenarios = "  ·  ".join(
        f"{_TC_IDS.get(t['scenario'], '??')} "
        f"{t['scenario'].replace('_', ' ').title()}"
        for t in trials[:cols])
    fig.suptitle(f"UHRC Navigation Trials — {scenarios}",
                 fontsize=TITLE_FS, fontweight="bold", color=TEXT_MAIN)
    fig.tight_layout()
    return fig


def draw_env_on_ax(ax, trial, label_prefix=""):
    """func to draw env panel onto existing Axes."""
    circles = trial["circles"]
    rects   = trial["rects"]
    traj    = trial["trajectory"]
    start   = trial["start"]
    goal    = trial["goal"]
    sc      = trial["scenario"]
    tc      = _TC_IDS.get(sc, "??")
    metrics = trial["metrics"]

    outcome = ("success"   if trial["success"]
               else "collision" if trial["collision"] else "timeout")
    tcolor  = TRAJ_COLOR[outcome]

    ax.set_facecolor(PANEL_BG)

    if circles or rects:
        clr      = build_clearance_grid(circles, rects)
        cmap_clr = LinearSegmentedColormap.from_list(
            "clr_light", ["#C8D6E5", "#EEF2F7", PANEL_BG])
        ax.imshow(clr, extent=[-10, 10, -10, 10], origin="lower",
                  cmap=cmap_clr, alpha=0.70, vmin=0.0, vmax=5.0,
                  aspect="equal", zorder=0)

    occ, ext = build_occupancy_grid(circles, rects)
    cmap_occ = LinearSegmentedColormap.from_list("occ_light", ["none", "#FFCDD2"])
    ax.imshow(occ, extent=ext, origin="lower", cmap=cmap_occ,
              alpha=0.55, vmin=0, vmax=1, aspect="equal", zorder=1)

    for (cx, cy, _), r in circles:
        ax.add_patch(Circle((cx, cy), r, lw=1.5,
                            edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))
    for cx, cy, hx, hy in rects:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, lw=1.5,
                                edgecolor=OBS_EDGE, facecolor=OBS_FACE, zorder=2))

    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=True,
                        facecolor=GOAL_FACE, zorder=3))
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=False,
                        ls="--", lw=1.4, edgecolor=GOAL_EDGE, zorder=3))

    if len(traj) > 1:
        for i in range(len(traj) - 1):
            alpha = 0.20 + 0.80 * (i / max(len(traj) - 2, 1))
            ax.plot(traj[i:i + 2, 0], traj[i:i + 2, 1],
                    color=tcolor, lw=1.8, alpha=alpha,
                    solid_capstyle="round", zorder=4)

    ax.plot(*start[:2], "o", ms=10, color="#16A34A",
            mec="white", mew=1.5, zorder=6)
    ax.plot(*goal[:2],  "*", ms=14, color="#F59E0B",
            mec="white", mew=1.0, zorder=6)
    ax.plot(*traj[-1, :2], "s", ms=9, color=tcolor,
            mec="white", mew=1.2, zorder=6)

    # Metric box
    metric_text = (
        f"RMSE : {metrics['rmse']:.3f} m\n"
        f"IAE  : {metrics['iae']:.3f} m·s\n"
        f"PLR  : {metrics['plr']:.3f}"
    )
    ax.text(0.98, 0.98, metric_text,
            transform=ax.transAxes,
            fontsize=METRIC_FS - 1, color=TEXT_MAIN, family="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.40", facecolor=BG,
                      edgecolor=GRID_COL, alpha=0.92),
            zorder=8)

    tc_str = f"{label_prefix}{tc} {sc.replace('_', ' ').title()}"
    ax.set_title(tc_str, fontsize=TITLE_FS - 1,
                 fontweight="bold", color=TEXT_MAIN, pad=5)
    ax.set_xlabel("$x$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_ylabel("$y$ (m)", fontsize=LABEL_FS, color=TEXT_SUB)
    ax.set_xlim(ARENA)
    ax.set_ylim(ARENA)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.5, alpha=0.6, color=GRID_COL)
    ax.tick_params(colors=TEXT_SUB, labelsize=TICK_FS)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    handles = [mpatches.Patch(color=tcolor, alpha=0.85,
                       label=f"{outcome.capitalize()}  "
                             f"(d = {trial['final_dist']:.2f} m)"), mpatches.Patch(color=OBS_EDGE, alpha=0.55, label="Obstacle"),mpatches.Patch(color=GOAL_EDGE, alpha=0.55,
                       label=f"Goal  (r = {GOAL_RADIUS} m)"), plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#16A34A", ms=9, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#F59E0B", ms=11, label="Goal"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    fontsize=LEGEND_FS - 1, framealpha=0.88,
                    edgecolor=GRID_COL, labelcolor=TEXT_MAIN, facecolor=BG)
    leg.set_draggable(True)



def parse_args():
    p = argparse.ArgumentParser(
        description="Report-quality UHRC trial plots — each panel in its own window")
    p.add_argument("--trial",      type=str,
                   default="benchmark_results/trials/open_field_trial001.npz",
                   help="Path to a single .npz trial file")
    p.add_argument("--scenario",   type=str, default=None,
                   help="Scenario name filter (use with --grid)")
    p.add_argument("--dir",        type=str,
                   default="benchmark_results/trials",
                   help="Directory containing .npz trial files")
    p.add_argument("--grid",       type=str, default=None,
                   help="Side-by-side grid of N trials (e.g. --grid 2)")
    p.add_argument("--env_only",   action="store_true",
                   help="Environment map only — for Methodology section")
    p.add_argument("--no_caption", action="store_true",
                   help="Suppress figure title / caption text")
    p.add_argument("--no_control", action="store_true",
                   help="Skip the control-input window")
    return p.parse_args()


def main():
    args  = parse_args()
    n_col = int(args.grid) if args.grid else 1

    if args.scenario:
        pattern = os.path.join(args.dir, f"{args.scenario}_trial*.npz")
        paths   = sorted(glob.glob(pattern))[:max(n_col, 1)]
    elif args.grid:
        paths   = sorted(glob.glob(os.path.join(args.dir, "*.npz")))[:n_col]
    else:
        paths   = [args.trial]

    if not paths:
        print("No trial files found ")
        return

    trials = [load_trial(p) for p in paths]

    for trial in trials:
        m  = trial["metrics"]
        sc = trial["scenario"]
        tc = _TC_IDS.get(sc, "??")
        print(f"\n{'─'*50}")
        print(f"  {tc}  |  {sc}  |  Trial {trial['trial_idx']}")
        outcome = ("SUCCESS"   if trial["success"]
                   else "COLLISION" if trial["collision"] else "TIMEOUT")
        print(f"  Outcome      : {outcome}")
        print(f"  Final dist   : {trial['final_dist']:.3f} m")
        print(f"  RMSE         : {m['rmse']:.3f} m")
        print(f"  IAE          : {m['iae']:.3f} m·s")
        print(f"  Path length  : {m['path_length']:.2f} m  "
              f"(straight = {m['straight']:.2f} m)")
        print(f"  PLR          : {m['plr']:.3f}")
        print(f"{'─'*50}")

    caption = not args.no_caption

    if args.env_only:
        plot_env_only(trials[0], caption=caption)

    elif args.grid and len(trials) > 1:
        # Grid view: one combined figure
        plot_grid(trials, cols=n_col)

    else:
        trial = trials[0]
        fig1 = plot_environment(trial, caption=caption)
        fig2 = plot_distance(trial,    caption=caption)
        if not args.no_control:
            fig3 = plot_control(trial, caption=caption)

    plt.show()


if __name__ == "__main__":
    main()