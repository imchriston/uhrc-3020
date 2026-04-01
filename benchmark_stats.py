"""
eval_uhrc_benchmark.py
======================
Structured Monte-Carlo benchmark for the UHRC controller.

Scenarios:
  TC-01  open_field        0 obstacles
  TC-02  sparse            2 circles
  TC-03  moderate          4 circles
  TC-04  dense             6 circles
  TC-05  narrow_corridor   4 circles arranged as pinch-point
  TC-06  gap_navigation    4 circles arranged as gap wall
  TC-07  cluttered         8 circles — stress test
  TC-08  close_range       2 circles, goal 3-5 m
  TC-09  long_range        4 circles, goal 12-18 m
  TC-10  urban_mixed       3 rect buildings + 2 circles
  TC-11  urban_dense       5 rect buildings
  TC-12  urban_corridor    rect buildings forming a street corridor

Per-trial output (saved to npz):
  trajectory, start, goal, obstacles_circles, obstacles_rects, metrics

Usage:
  python benchmark_stats.py
  python benchmark_stats.py --scenario urban_mixed --n 20 --verbose
  python benchmark_stats.py --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle

import drone.dynamics as dynamics
from generate_data import get_lidar_scan
from uhrc_ctrl import UHRCController
import utils.quat_euler as quat_euler

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH      = "checkpoints/uhrc_best.pth"
STATS_PATH      = "checkpoints/norm_stats.npz"
DT              = 0.01
MAX_STEPS       = 1500
GOAL_RADIUS     = 0.5
CONVERGE_THRESH = 1.0
HOVER_THRUST    = 9.81
ARENA           = (-10.0, 10.0)
DEFAULT_TRIALS  = 30

OBS_COUNT = {
    "open_field":       0,
    "sparse":           2,
    "moderate":         4,
    "dense":            6,
    "narrow_corridor":  4,
    "gap_navigation":   4,
    "cluttered":        8,
    "close_range":      2,
    "long_range":       4,
    "urban_mixed":      5,      
    "urban_dense":      5,      
    "urban_corridor":   6,      
}

_TC_IDS = {
    "open_field":      "TC-01", "sparse":         "TC-02",
    "moderate":        "TC-03", "dense":          "TC-04",
    "narrow_corridor": "TC-05", "gap_navigation": "TC-06",
    "cluttered":       "TC-07", "close_range":    "TC-08",
    "long_range":      "TC-09", "urban_mixed":    "TC-10",
    "urban_dense":     "TC-11", "urban_corridor": "TC-12",
}


# ══════════════════════════════════════════════════════════════════════════════
#  RECTANGULAR BUILDING SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RectBuilding:
    """Axis-aligned rectangular obstacle (building)."""
    cx: float
    cy: float
    hx: float    # half-width in x
    hy: float    # half-height in y

    @property
    def xmin(self): return self.cx - self.hx
    @property
    def xmax(self): return self.cx + self.hx
    @property
    def ymin(self): return self.cy - self.hy
    @property
    def ymax(self): return self.cy + self.hy

    def contains(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def to_array(self):
        """[cx, cy, hx, hy] for serialisation."""
        return np.array([self.cx, self.cy, self.hx, self.hy], dtype=np.float32)


def _ray_aabb_2d(px, py, dx, dy, bld: RectBuilding) -> float:
    """Ray-AABB 2D intersection.  Returns t_hit or inf."""
    eps = 1e-9
    tmin, tmax = -1e18, 1e18
    if abs(dx) < eps:
        if px < bld.xmin or px > bld.xmax: return float("inf")
    else:
        tx1 = (bld.xmin - px) / dx
        tx2 = (bld.xmax - px) / dx
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))
    if abs(dy) < eps:
        if py < bld.ymin or py > bld.ymax: return float("inf")
    else:
        ty1 = (bld.ymin - py) / dy
        ty2 = (bld.ymax - py) / dy
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
    if tmax < 0.0 or tmin > tmax:
        return float("inf")
    t_hit = tmin if tmin >= 0.0 else tmax
    return t_hit if t_hit >= 0.0 else float("inf")


def lidar_scan_mixed(pos, yaw, circles, rects,
                     num_rays=32, fov=np.pi, max_range=5.0):
    """
    Unified lidar: handles both circular and rectangular obstacles.
    Circles: list of ((cx,cy,cz), r)
    Rects:   list of RectBuilding
    Returns ranges [N] matching get_lidar_scan signature.
    """
    ranges = get_lidar_scan(pos, yaw, circles, num_rays=num_rays,
                            fov=fov, max_range=max_range)

    if not rects:
        return ranges

    # Cast rectangles on top
    ray_angles = np.linspace(-0.5 * fov, 0.5 * fov, num_rays, dtype=np.float64)
    global_angles = yaw + ray_angles
    dirs = np.stack([np.cos(global_angles), np.sin(global_angles)], axis=1)
    px, py = float(pos[0]), float(pos[1])

    for bld in rects:
        for i in range(num_rays):
            t = _ray_aabb_2d(px, py, float(dirs[i, 0]), float(dirs[i, 1]), bld)
            if t < ranges[i]:
                ranges[i] = float(t)

    return np.minimum(ranges, max_range).astype(np.float32)


def rect_collision(r_I, rects):
    """Check if drone position is inside any rectangle."""
    x, y = float(r_I[0]), float(r_I[1])
    return any(b.contains(x, y) for b in rects)


def rect_clearance(r_I, bld: RectBuilding) -> float:
    """Signed distance from point to rectangle boundary (positive = outside)."""
    x, y = float(r_I[0]), float(r_I[1])
    dx = max(bld.xmin - x, 0.0, x - bld.xmax)
    dy = max(bld.ymin - y, 0.0, y - bld.ymax)
    return float(np.sqrt(dx*dx + dy*dy))


#  OBSTACLE BUILDERS

def _min_sep(centers, cand, radii, r_c, margin=0.3):
    for c, r in zip(centers, radii):
        if np.linalg.norm(np.array(c) - np.array(cand)) < r + r_c + margin:
            return False
    return True

def _safe_obs(cx, cy, r, start, goal, gap=0.8):
    for pt in (start[:2], goal[:2]):
        if np.linalg.norm(np.array([cx, cy]) - np.array(pt)) < r + gap:
            return False
    return True

def _safe_rect(bld: RectBuilding, start, goal, gap=1.0):
    """Check rectangle (inflated by gap) doesn't cover start or goal."""
    for pt in (start[:2], goal[:2]):
        inf = RectBuilding(bld.cx, bld.cy, bld.hx + gap, bld.hy + gap)
        if inf.contains(float(pt[0]), float(pt[1])):
            return False
    return True

def build_circles_random(n, start, goal, rng):
    obs, centers, radii = [], [], []
    for _ in range(500):
        if len(obs) >= n: break
        cx = float(rng.uniform(*ARENA))
        cy = float(rng.uniform(*ARENA))
        r  = float(rng.uniform(0.5, 1.2))
        if not _safe_obs(cx, cy, r, start, goal): continue
        if not _min_sep(centers, (cx, cy), radii, r): continue
        obs.append(((cx, cy, 0.0), r))
        centers.append((cx, cy)); radii.append(r)
    return obs

def build_rects_random(n, start, goal, rng):
    """Random axis-aligned rectangular buildings."""
    rects = []
    for _ in range(500):
        if len(rects) >= n: break
        cx = float(rng.uniform(*ARENA))
        cy = float(rng.uniform(*ARENA))
        hx = float(rng.uniform(0.5, 1.5))
        hy = float(rng.uniform(0.5, 1.5))
        bld = RectBuilding(cx, cy, hx, hy)
        if not _safe_rect(bld, start, goal): continue
        # Check no overlap with existing rects
        overlap = False
        for existing in rects:
            if (abs(cx - existing.cx) < hx + existing.hx + 0.5 and
                abs(cy - existing.cy) < hy + existing.hy + 0.5):
                overlap = True; break
        if overlap: continue
        rects.append(bld)
    return rects

def build_obstacles_narrow_corridor(start, goal, rng):
    mid   = (start[:2] + goal[:2]) / 2.0
    perp  = np.array([-(goal[1]-start[1]), goal[0]-start[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    fwd   = (goal[:2]-start[:2]); fwd /= np.linalg.norm(fwd)+1e-9
    r = 0.7
    gh = 0.8 + float(rng.uniform(0.0, 0.4))
    fo = float(rng.uniform(-1.5, 1.5))
    obs = []
    for s1, s2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        c = mid + perp*s1*(gh+r) + fwd*s2*fo
        if ARENA[0]+1 < c[0] < ARENA[1]-1 and ARENA[0]+1 < c[1] < ARENA[1]-1:
            obs.append(((float(c[0]), float(c[1]), 0.0), r))
    while len(obs) < 4:
        obs += build_circles_random(1, start, goal, rng)
    return obs[:4]

def build_obstacles_gap(start, goal, rng):
    mid  = (start[:2]+goal[:2])/2.0
    perp = np.array([-(goal[1]-start[1]), goal[0]-start[0]])
    perp /= np.linalg.norm(perp)+1e-9
    fwd  = (goal[:2]-start[:2]); fwd /= np.linalg.norm(fwd)+1e-9
    r = 0.9; gp = float(rng.uniform(-0.5, 0.5))
    obs = []
    for offset in [2.1*r, -2.1*r, 5.0*r]:
        c = mid + perp*(gp + offset)
        if ARENA[0]+1 < c[0] < ARENA[1]-1 and ARENA[0]+1 < c[1] < ARENA[1]-1:
            obs.append(((float(c[0]), float(c[1]), 0.0), r))
    c_d = mid + fwd*2.0 + perp*float(rng.uniform(-1.5, 1.5))
    obs.append(((float(c_d[0]), float(c_d[1]), 0.0), r))
    while len(obs) < 4:
        obs += build_circles_random(1, start, goal, rng)
    return obs[:4]

def build_urban_corridor(start, goal, rng):
    """Rectangular buildings forming a street corridor the drone flies through."""
    fwd  = goal[:2] - start[:2]
    fwd /= np.linalg.norm(fwd) + 1e-9
    perp = np.array([-fwd[1], fwd[0]])
    d    = np.linalg.norm(goal[:2] - start[:2])

    rects = []
    n_per_side = 3
    spacing = d / (n_per_side + 1)
    corridor_half = 2.0 + float(rng.uniform(0.0, 0.5))

    for i in range(1, n_per_side + 1):
        along = start[:2] + fwd * (i * spacing)
        jitter = float(rng.uniform(-0.3, 0.3))
        for side in [1, -1]:
            cx = float(along[0] + perp[0] * (corridor_half + 0.8) * side)
            cy = float(along[1] + perp[1] * (corridor_half + 0.8) * side + jitter)
            hx = float(rng.uniform(0.6, 1.2))
            hy = float(rng.uniform(0.6, 1.2))
            bld = RectBuilding(cx, cy, hx, hy)
            if _safe_rect(bld, start, goal, gap=0.8):
                rects.append(bld)
    return rects[:6]


def _sample_start_goal(scenario, rng):
    lo, hi = ARENA
    if scenario == "close_range":
        start = np.array([float(rng.uniform(lo, hi)),
                          float(rng.uniform(lo, hi)), 0.0])
        for _ in range(200):
            a = float(rng.uniform(0, 2*np.pi))
            d = float(rng.uniform(3.0, 5.0))
            goal = start + np.array([np.cos(a)*d, np.sin(a)*d, 0.0])
            if lo < goal[0] < hi and lo < goal[1] < hi:
                return start, goal
        return start, start + np.array([4.0, 0.0, 0.0])
    elif scenario == "long_range":
        for _ in range(500):
            s = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
            g = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
            if 12.0 <= np.linalg.norm(g[:2]-s[:2]) <= 18.0:
                return s, g
        return np.array([-9.,  -9., 0.]), np.array([9., 9., 0.])
    else:
        start = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
        for _ in range(200):
            goal = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
            if np.linalg.norm(goal[:2]-start[:2]) >= 3.0:
                return start, goal
        return start, start + np.array([6.0, 0.0, 0.0])


#  DATA STRUCTURES

@dataclass
class TrialResult:
    scenario:          str
    trial_idx:         int
    seed:              int
    n_obstacles:       int
    goal_dist_init:    float
    success:           bool
    collision:         bool
    timeout:           bool
    final_dist:        float
    min_dist:          float
    convergence_step:  int
    steps_taken:       int
    path_length:       float
    mean_dist:         float
    fz_mean:           float
    fz_max:            float
    fz_drift:          float
    min_obs_clearance: float
    wall_time_s:       float

@dataclass
class ScenarioStats:
    scenario:              str
    n_trials:              int
    n_obstacles:           int
    success_rate:          float
    collision_rate:        float
    timeout_rate:          float
    mean_final_dist:       float
    std_final_dist:        float
    mean_min_dist:         float
    std_min_dist:          float
    mean_convergence_step: float
    mean_path_length:      float
    std_path_length:       float
    mean_fz_drift:         float
    std_fz_drift:          float
    mean_obs_clearance:    float
    std_obs_clearance:     float
    mean_wall_time_s:      float



#  SINGLE TRIAL RUNNER

def _rk4_step(dyn, t, x, u, dt=DT):
    def f(tt, xx): return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t, x)
    k2 = f(t+.5*dt, x+.5*dt*k1)
    k3 = f(t+.5*dt, x+.5*dt*k2)
    k4 = f(t+dt,    x+dt*k3)
    xn = x + (dt/6.)*(k1+2*k2+2*k3+k4)
    q = xn[6:10]; xn[6:10] = q/(np.linalg.norm(q)+1e-12)
    return xn


def run_trial(scenario, trial_idx, seed, verbose=False, save_dir=None):
    """
    Execute one episode.  Returns (TrialResult, trial_data_dict).
    trial_data_dict contains trajectory/obstacles/start/goal for replay.
    """
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    params = dynamics.QuadrotorParams()
    dyn    = dynamics.QuadrotorDynamics(params)
    ctrl   = UHRCController(MODEL_PATH, STATS_PATH, device="cpu")
    ctrl.reset()

    start, goal = _sample_start_goal(scenario, rng)

    # Build obstacles
    circles = []
    rects   = []

    if scenario == "narrow_corridor":
        circles = build_obstacles_narrow_corridor(start, goal, rng)
    elif scenario == "gap_navigation":
        circles = build_obstacles_gap(start, goal, rng)
    elif scenario == "urban_mixed":
        rects   = build_rects_random(3, start, goal, rng)
        circles = build_circles_random(2, start, goal, rng)
    elif scenario == "urban_dense":
        rects   = build_rects_random(5, start, goal, rng)
    elif scenario == "urban_corridor":
        rects   = build_urban_corridor(start, goal, rng)
    else:
        circles = build_circles_random(OBS_COUNT.get(scenario, 0),
                                       start, goal, rng)

    n_total_obs    = len(circles) + len(rects)
    has_rects      = len(rects) > 0
    goal_dist_init = float(np.linalg.norm(goal[:2] - start[:2]))

    x_curr = dyn.pack_state(
        start, np.zeros(3), np.array([1., 0., 0., 0.]),
        np.zeros(3), np.zeros(4),
    )

    positions       = [start.copy()]
    dists           = []
    fz_log          = []
    tau_phi_log     = []   # U2 roll torque  (N·m)
    tau_theta_log   = []   # U3 pitch torque (N·m)
    tau_psi_log     = []   # U4 yaw torque   (N·m)
    subgoal_log     = []
    obs_clear_log   = []
    t_sim         = 0.0
    success = collision = False
    conv_step = -1

    for step in range(MAX_STEPS):
        r_I, v_I, q_BI, w_B, Omega = dyn.unpack_state(x_curr)
        psi = float(quat_euler.euler_from_q(q_BI)[2])

        # Unified lidar for circles + rects
        if has_rects:
            lidar = lidar_scan_mixed(r_I, psi, circles, rects)
        else:
            lidar = get_lidar_scan(r_I, psi, circles, num_rays=32)

        u_nn, sub_nn = ctrl.get_action(r_I, v_I, q_BI, w_B, Omega, lidar, goal)
        fz_log.append(float(u_nn[0]))        # U1  total thrust   (N)
        tau_phi_log.append(float(u_nn[1]))    # U2  roll torque    (N·m)
        tau_theta_log.append(float(u_nn[2]))  # U3  pitch torque   (N·m)
        tau_psi_log.append(float(u_nn[3]))    # U4  yaw torque     (N·m)
        subgoal_log.append(sub_nn.copy())

        dist = float(np.linalg.norm(r_I[:2] - goal[:2]))
        dists.append(dist)
        if conv_step == -1 and dist < CONVERGE_THRESH:
            conv_step = step

        # Clearance to ensure obstacle is not spawned on top of drone at start or ar the goal
        clearances = []
        for c, r_obs in circles:
            clearances.append(
                float(np.linalg.norm(r_I[:2] - np.asarray(c[:2]))) - float(r_obs))
        for bld in rects:
            clearances.append(rect_clearance(r_I, bld))
        if clearances:
            obs_clear_log.append(min(clearances))

        if verbose and step % 100 == 0:
            print(f"  [{scenario}] trial {trial_idx} step {step:4d}"
                  f"  pos=({r_I[0]:6.2f},{r_I[1]:6.2f})"
                  f"  dist={dist:.2f}m  Fz={u_nn[0]:.3f}"
                  f"  sub=({sub_nn[0]:+.2f},{sub_nn[1]:+.2f})")

        x_curr = _rk4_step(dyn, t_sim, x_curr, u_nn)
        t_sim += DT
        r_new, *_ = dyn.unpack_state(x_curr)
        positions.append(r_new.copy())

        # Collision: circles
        hit_circle = any(
            float(np.linalg.norm(r_new[:2] - np.asarray(c[:2]))) < float(r_obs)
            for c, r_obs in circles
        )
        hit_rect = rect_collision(r_new, rects)

        reached = float(np.linalg.norm(r_new[:2] - goal[:2])) < GOAL_RADIUS

        if hit_circle or hit_rect:
            collision = True; break
        if reached:
            success = True; break

    steps_taken = len(positions) - 1
    path = np.array(positions)
    path_length = float(np.sum(np.linalg.norm(
        np.diff(path[:, :2], axis=0), axis=1)))
    final_dist = float(np.linalg.norm(path[-1, :2] - goal[:2]))
    min_dist   = float(min(dists)) if dists else goal_dist_init
    mean_dist  = float(np.mean(dists)) if dists else goal_dist_init

    fz_arr   = np.array(fz_log) if fz_log else np.array([HOVER_THRUST])
    fz_mean  = float(fz_arr.mean())
    fz_max   = float(fz_arr.max())
    fz_drift = float(np.mean(np.abs(fz_arr - HOVER_THRUST)))
    min_obs_clearance = float(min(obs_clear_log)) if obs_clear_log else -1.0

    wall_time = time.perf_counter() - t0

    result = TrialResult(
        scenario=scenario, trial_idx=trial_idx, seed=seed,
        n_obstacles=n_total_obs, goal_dist_init=goal_dist_init,
        success=success, collision=collision,
        timeout=not success and not collision,
        final_dist=final_dist, min_dist=min_dist,
        convergence_step=conv_step, steps_taken=steps_taken,
        path_length=path_length, mean_dist=mean_dist,
        fz_mean=fz_mean, fz_max=fz_max, fz_drift=fz_drift,
        min_obs_clearance=min_obs_clearance, wall_time_s=wall_time,
    )

    # ── Pack trial geometry for saving ────────────────────────────────────
    # Circles: [N, 3] → (cx, cy, r)
    if circles:
        circ_arr = np.array([[c[0], c[1], r] for c, r in circles],
                            dtype=np.float32)
    else:
        circ_arr = np.zeros((0, 3), dtype=np.float32)

    # Rects: [M, 4] → (cx, cy, hx, hy)
    if rects:
        rect_arr = np.array([b.to_array() for b in rects], dtype=np.float32)
    else:
        rect_arr = np.zeros((0, 4), dtype=np.float32)

    # outcome_code: 1 = success, 2 = collision, 0 = timeout
    outcome_code = np.int8(1 if success else (2 if collision else 0))

    trial_data = {
        "trajectory":      path.astype(np.float32),            # [T+1, 3]
        "start":           start.astype(np.float32),           # [3]
        "goal":            goal.astype(np.float32),            # [3]
        "circles":         circ_arr,                           # [N, 3] cx cy r
        "rects":           rect_arr,                           # [M, 4] cx cy hx hy
        "outcome_code":    np.array([outcome_code], dtype=np.int8),  # 1=success 2=collision 0=timeout
        "success":         np.array([success],   dtype=bool),
        "collision":       np.array([collision], dtype=bool),
        "timeout":         np.array([not success and not collision], dtype=bool),
        "final_dist":      np.array([final_dist], dtype=np.float32),
        "seed":            np.array([seed],       dtype=np.int64),
        #Control inputs U1–U4 
        "fz_log":          fz_arr.astype(np.float32),          #   U1 thrust (N)
        "tau_phi_log":     np.array(tau_phi_log,   dtype=np.float32),  #  U2 roll  (N·m)
        "tau_theta_log":   np.array(tau_theta_log, dtype=np.float32),  #  U3 pitch (N·m)
        "tau_psi_log":     np.array(tau_psi_log,   dtype=np.float32),  #  U4 yaw   (N·m)
        # Navigation logs 
        "dist_log":        np.array(dists,        dtype=np.float32),   
        "subgoal_log":     np.array(subgoal_log,  dtype=np.float32),   
    }

    # Save per-trial npz 
    if save_dir is not None:
        trial_path = os.path.join(save_dir, "trials",
                                  f"{scenario}_trial{trial_idx:03d}.npz")
        os.makedirs(os.path.dirname(trial_path), exist_ok=True)
        np.savez_compressed(trial_path, **trial_data)

    return result, trial_data


#  AGGREGATE OUTPUT

def compute_stats(trials):
    sc = trials[0].scenario; n = len(trials)
    n_obs = trials[0].n_obstacles
    fd = [t.final_dist for t in trials]
    md = [t.min_dist for t in trials]
    pl = [t.path_length for t in trials]
    fzd = [t.fz_drift for t in trials]
    wt = [t.wall_time_s for t in trials]
    clr = [t.min_obs_clearance for t in trials if t.min_obs_clearance >= 0]
    cvk = [t.convergence_step for t in trials if t.convergence_step >= 0]
    return ScenarioStats(
        scenario=sc, n_trials=n, n_obstacles=n_obs,
        success_rate=float(np.mean([t.success for t in trials])),
        collision_rate=float(np.mean([t.collision for t in trials])),
        timeout_rate=float(np.mean([t.timeout for t in trials])),
        mean_final_dist=float(np.mean(fd)), std_final_dist=float(np.std(fd)),
        mean_min_dist=float(np.mean(md)), std_min_dist=float(np.std(md)),
        mean_convergence_step=float(np.mean(cvk)) if cvk else float("nan"),
        mean_path_length=float(np.mean(pl)), std_path_length=float(np.std(pl)),
        mean_fz_drift=float(np.mean(fzd)), std_fz_drift=float(np.std(fzd)),
        mean_obs_clearance=float(np.mean(clr)) if clr else float("nan"),
        std_obs_clearance=float(np.std(clr)) if clr else float("nan"),
        mean_wall_time_s=float(np.mean(wt)),
    )


def print_summary_table(stats_list):
    col_w = [8, 18, 5, 8, 8, 8, 12, 12, 12, 10]
    header = ["TC-ID", "Scenario", "Obs", "Succ%", "Coll%", "Tout%",
              "FinalDist", "MinDist", "PathLen", "FzDrift"]
    sep = "\u2500" * (sum(col_w) + 3*len(col_w))
    print("\n" + sep)
    print("  UHRC Benchmark Results")
    print(sep)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)
    for s in stats_list:
        tc = _TC_IDS.get(s.scenario, "TC-??")
        row = [tc, s.scenario, str(s.n_obstacles),
               f"{s.success_rate*100:.1f}", f"{s.collision_rate*100:.1f}",
               f"{s.timeout_rate*100:.1f}",
               f"{s.mean_final_dist:.2f}\u00b1{s.std_final_dist:.2f}",
               f"{s.mean_min_dist:.2f}\u00b1{s.std_min_dist:.2f}",
               f"{s.mean_path_length:.2f}\u00b1{s.std_path_length:.2f}",
               f"{s.mean_fz_drift:.3f}\u00b1{s.std_fz_drift:.3f}"]
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, col_w)))
    print(sep)


def save_results(trials, stats, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Per-trial CSV
    td = [asdict(t) for t in trials]
    if td:
        with open(os.path.join(out_dir, "trials_wp.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=td[0].keys())
            w.writeheader(); w.writerows(td)
    # Summary CSV
    sd = [asdict(s) for s in stats]
    if sd:
        with open(os.path.join(out_dir, "summary_wp.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sd[0].keys())
            w.writeheader(); w.writerows(sd)
    # JSON
    with open(os.path.join(out_dir, "results_wp.json"), "w") as f:
        json.dump({"trials": td, "summary": sd}, f, indent=2)
    print(f"  Saved CSV + JSON → {out_dir}/")


def plot_summary(stats_list, out_dir):
    scenarios = [s.scenario.replace("_", "\n") for s in stats_list]
    x = np.arange(len(scenarios)); bw = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("UHRC Benchmark \u2014 Scenario Performance Summary",
                 fontsize=13, y=0.98)

    ax = axes[0, 0]
    succ = [s.success_rate*100 for s in stats_list]
    coll = [s.collision_rate*100 for s in stats_list]
    tout = [s.timeout_rate*100 for s in stats_list]
    ax.bar(x, succ, bw, label="Success", color="#4CAF50")
    ax.bar(x, coll, bw, bottom=succ, label="Collision", color="#F44336")
    btm = [a+b for a,b in zip(succ, coll)]
    ax.bar(x, tout, bw, bottom=btm, label="Timeout", color="#FFC107")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=6)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Outcome breakdown"); ax.legend(fontsize=7)

    ax = axes[0, 1]
    mfd = [s.mean_final_dist for s in stats_list]
    sfd = [s.std_final_dist for s in stats_list]
    ax.bar(x, mfd, bw, color="#2196F3", alpha=0.85)
    ax.errorbar(x, mfd, yerr=sfd, fmt="none", color="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=6)
    ax.set_ylabel("m"); ax.set_title("Final distance (mean\u00b1std)")
    ax.axhline(GOAL_RADIUS, color="#4CAF50", ls="--", label=f"Goal {GOAL_RADIUS}m")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    mpl = [s.mean_path_length for s in stats_list]
    spl = [s.std_path_length for s in stats_list]
    ax.bar(x, mpl, bw, color="#9C27B0", alpha=0.85)
    ax.errorbar(x, mpl, yerr=spl, fmt="none", color="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=6)
    ax.set_ylabel("m"); ax.set_title("Path length (mean\u00b1std)")

    ax = axes[1, 1]
    fzd = [s.mean_fz_drift for s in stats_list]
    sfd2 = [s.std_fz_drift for s in stats_list]
    ax.bar(x, fzd, bw, color="#FF9800", alpha=0.85)
    ax.errorbar(x, fzd, yerr=sfd2, fmt="none", color="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=6)
    ax.set_ylabel("N"); ax.set_title("Fz drift (mean |Fz\u22129.81|)")
    ax.axhline(0.5, color="#F44336", ls="--", label="0.5 N")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_summary_wp.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trial(trial_data, result, out_path):
    """Plot a single trial: trajectory + obstacles + occupancy context."""
    fig, ax = plt.subplots(figsize=(10, 8))

    traj = trial_data["trajectory"]
    start = trial_data["start"]
    goal  = trial_data["goal"]

    # Draw circles
    for i in range(len(trial_data["circles"])):
        cx, cy, r = trial_data["circles"][i]
        ax.add_patch(Circle((cx, cy), r, color="tomato", alpha=0.5))

    # Draw rectangles 
    for i in range(len(trial_data["rects"])):
        cx, cy, hx, hy = trial_data["rects"][i]
        ax.add_patch(Rectangle(
            (cx - hx, cy - hy), 2*hx, 2*hy,
            linewidth=2, edgecolor="darkred",
            facecolor="salmon", alpha=0.6))

    # Trajectory
    color = "royalblue" if result.success else ("red" if result.collision else "orange")
    label = "Success" if result.success else ("Collision" if result.collision else "Timeout")
    ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2, label=f"UHRC ({label})")
    ax.plot(start[0], start[1], "go", ms=10, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "b*", ms=14, label="Goal", zorder=5)
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS, fill=False,
                        ls="--", color="blue", alpha=0.4))

    ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{_TC_IDS.get(result.scenario, '??')} {result.scenario}  "
                 f"seed={result.seed}  dist={result.final_dist:.2f}m  "
                 f"Fz_drift={result.fz_drift:.3f}N")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


#  MAIN

def parse_args():
    p = argparse.ArgumentParser(description="UHRC benchmark")
    p.add_argument("--scenario", type=str, default=None)
    p.add_argument("--n", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--out", type=str, default="benchmark_results")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--plot_trials", action="store_true",
                   help="Save a trajectory plot for every trial")
    return p.parse_args()


def main():
    args = parse_args()
    all_scenarios = list(OBS_COUNT.keys())

    if args.quick:
        scenarios = all_scenarios[:3]; n_trials = 10
    elif args.scenario:
        if args.scenario not in OBS_COUNT:
            print(f"Unknown: '{args.scenario}'. Choose from: {all_scenarios}")
            return
        scenarios = [args.scenario]; n_trials = args.n
    else:
        scenarios = all_scenarios; n_trials = args.n

    print(f"\n{'='*62}")
    print(f"  UHRC Benchmark  |  {len(scenarios)} scenario(s)  |  {n_trials} trials each")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Out   : {args.out}/")
    print(f"{'='*62}\n")

    all_trials  = []
    all_stats   = []
    grand_total = len(scenarios) * n_trials
    done = 0

    for sc in scenarios:
        tc = _TC_IDS.get(sc, "??")
        print(f"  \u25b6 {tc} {sc}  (obs={OBS_COUNT[sc]}, n={n_trials})")
        sc_trials = []

        for i in range(n_trials):
            seed = args.seed_base + done
            tr, td = run_trial(sc, i, seed,
                               verbose=args.verbose,
                               save_dir=args.out)
            sc_trials.append(tr)
            done += 1

            icon = "\u2705" if tr.success else ("\U0001f4a5" if tr.collision else "\u23f1")
            print(f"    [{done:4d}/{grand_total}] trial {i+1:3d}/{n_trials}"
                  f"  {icon}  dist={tr.final_dist:5.2f}m"
                  f"  steps={tr.steps_taken:4d}"
                  f"  Fz_drift={tr.fz_drift:.3f}N", flush=True)

            if args.plot_trials:
                fig_dir = os.path.join(args.out, "plots")
                os.makedirs(fig_dir, exist_ok=True)
                plot_trial(td, tr,
                           os.path.join(fig_dir,
                                        f"{sc}_trial{i:03d}.png"))

        stats = compute_stats(sc_trials)
        all_trials.extend(sc_trials)
        all_stats.append(stats)
        print(f"    \u2192 succ={stats.success_rate*100:.1f}%  "
              f"coll={stats.collision_rate*100:.1f}%  "
              f"tout={stats.timeout_rate*100:.1f}%  "
              f"mean_dist={stats.mean_final_dist:.2f}m\n")

    print_summary_table(all_stats)
    save_results(all_trials, all_stats, args.out)
    plot_summary(all_stats, args.out)

    total_succ = sum(t.success for t in all_trials)
    print(f"\n  Overall: {total_succ}/{len(all_trials)} "
          f"({total_succ/len(all_trials)*100:.1f}%)\n")


if __name__ == "__main__":
    main()