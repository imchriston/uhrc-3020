"""
eval_uhrc_dynamic.py
====================
Dynamic-obstacle Monte-Carlo benchmark for the UHRC waypoint controller.

Scenario: Random-timed obstacle pop-in (TC-D01).
  Circular obstacles appear at uniformly-sampled random times after the
  drone is already in flight, testing real-time reactive avoidance.

Controller architecture (identical to eval_uhrc_benchmark.py):
  UHRCWaypointController.get_att_ref() → [phi_ref, theta_ref]  (outer loop)
  AttitudePID.step(refs)               → [Fz, τx, τy, τz]      (inner loop)

Performance metrics:
  RMSE, IAE, ITAE, PLR  — same integral error metrics as static benchmark
  near_miss_count        — steps where min-clearance < NEAR_MISS_THRESH (0.3 m)
  reaction_step          — first step heading changed after an obstacle appeared
  dynamic_obs_hit        — whether any collision involved a dynamic obstacle

Usage:
  python eval_uhrc_dynamic.py
  python eval_uhrc_dynamic.py --n 30 --verbose
  python eval_uhrc_dynamic.py --n_obs 3 --quick
  python eval_uhrc_dynamic.py --plot_trials
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

import dynamics
import controller.pid as pid
import controller.attitude as angle_control
from generate_data_sensors import get_lidar_scan
from uhrc_ctrl_wp import UHRCWaypointController
import utils.quat_euler as quat_euler

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/uhrc_best_waypoints.pth"
STATS_PATH = "checkpoints/norm_stats_waypoints.npz"

# ── Simulation ────────────────────────────────────────────────────────────────
DT               = 0.01
MAX_STEPS        = 3000
GOAL_RADIUS      = 0.5
CONVERGE_THRESH  = 1.0
HOVER_Z          = 1.0
ARENA            = (-10.0, 10.0)
DEFAULT_TRIALS   = 20
NEAR_MISS_THRESH = 0.3   # metres
LIDAR_MAX        = 5.0

# ── Dynamic obstacle defaults ─────────────────────────────────────────────────
DEFAULT_N_OBS    = 3      # number of obstacles that pop in per episode
OBS_RADIUS_RANGE = (0.5, 1.0)
# Pop timing: obstacles appear at random steps uniformly sampled in this
# fraction-of-MAX_STEPS range so the drone always has time to react.
POP_WINDOW_FRAC  = (0.15, 0.65)

# ── AttitudePID gains — must exactly match generate_data_sensors.run() ────────
_GAINS_ROLL  = pid.PIDGains(kp=4.1602, ki=0.0,  kd=2.0247)
_GAINS_PITCH = pid.PIDGains(kp=4.1602, ki=0.0,  kd=2.0247)
_GAINS_YAW   = pid.PIDGains(kp=0.9848, ki=0.0,  kd=0.9542)
_GAINS_Z     = pid.PIDGains(kp=153.0,  ki=61.0, kd=135.0)
MAX_TILT_RAD = math.radians(30.0)

SCENARIO_NAME = "dyn_random_pop"
TC_ID         = "TC-D01"


# ══════════════════════════════════════════════════════════════════════════════
#  ATTITUDE PID FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_att_ctrl(params: dynamics.QuadrotorParams) -> angle_control.AttitudePID:
    return angle_control.AttitudePID(
        params          = params,
        gains_roll      = _GAINS_ROLL,
        gains_pitch     = _GAINS_PITCH,
        gains_yaw       = _GAINS_YAW,
        gains_z         = _GAINS_Z,
        torque_limits   = (0.6, 0.6, 0.35),
        thrust_limits   = (0.0, None),
        az_limit        = 6.0,
        d_cut_angles_hz = 10.0,
        d_cut_vz_hz     = 5.0,
        sample_time     = DT,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE PLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _sample_start_goal(rng, min_dist=5.0, max_dist=14.0):
    lo, hi = ARENA
    for _ in range(500):
        s = np.array([float(rng.uniform(lo, hi)),
                      float(rng.uniform(lo, hi)), HOVER_Z])
        g = np.array([float(rng.uniform(lo, hi)),
                      float(rng.uniform(lo, hi)), HOVER_Z])
        if min_dist <= np.linalg.norm(g[:2]-s[:2]) <= max_dist:
            return s, g
    return np.array([-7., 0., HOVER_Z]), np.array([7., 0., HOVER_Z])


@dataclass
class PopObstacle:
    """A circular obstacle with a fixed position that appears at appear_step."""
    cx: float
    cy: float
    r:  float
    appear_step: int

    def is_active(self, step: int) -> bool:
        return step >= self.appear_step

    def to_circle_tuple(self) -> Tuple:
        return ((self.cx, self.cy, 0.0), self.r)


def _safe_placement(cx, cy, r, start, goal, existing, gap=1.0):
    """Check obstacle doesn't overlap start/goal/existing obstacles."""
    for pt in (start[:2], goal[:2]):
        if np.linalg.norm(np.array([cx, cy])-pt) < r + gap:
            return False
    for obs in existing:
        if np.linalg.norm(np.array([cx, cy])-np.array([obs.cx, obs.cy])) < r + obs.r + 0.3:
            return False
    return True


def build_pop_obstacles(n: int, start: np.ndarray, goal: np.ndarray,
                        rng) -> List[PopObstacle]:
    """
    Place n obstacles at random positions along the flight corridor and
    assign each a uniformly-sampled pop-in step.

    Placement: obstacles are sampled along the start→goal axis (fraction
    0.2–0.85 of the path) with small lateral jitter, so they realistically
    threaten the drone's path.
    """
    fwd  = goal[:2] - start[:2]
    dist = float(np.linalg.norm(fwd)) + 1e-9
    fwd  = fwd / dist
    perp = np.array([-fwd[1], fwd[0]])

    pop_lo = int(POP_WINDOW_FRAC[0] * MAX_STEPS)
    pop_hi = int(POP_WINDOW_FRAC[1] * MAX_STEPS)

    obstacles: List[PopObstacle] = []
    for _ in range(500):
        if len(obstacles) >= n:
            break
        t_frac  = float(rng.uniform(0.20, 0.85))
        lat     = float(rng.uniform(-1.5, 1.5))
        r       = float(rng.uniform(*OBS_RADIUS_RANGE))
        cx = float(start[0] + fwd[0]*dist*t_frac + perp[0]*lat)
        cy = float(start[1] + fwd[1]*dist*t_frac + perp[1]*lat)
        if not (ARENA[0] < cx < ARENA[1] and ARENA[0] < cy < ARENA[1]):
            continue
        if not _safe_placement(cx, cy, r, start, goal, obstacles):
            continue
        appear_step = int(rng.integers(pop_lo, pop_hi))
        obstacles.append(PopObstacle(cx=cx, cy=cy, r=r, appear_step=appear_step))

    return obstacles


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_integral_metrics(dists: list, dt: float):
    d    = np.array(dists, dtype=np.float64)
    t    = np.arange(len(d)) * dt
    rmse = float(np.sqrt(np.mean(d**2)))
    iae  = float(np.sum(np.abs(d)) * dt)
    itae = float(np.sum(t * np.abs(d)) * dt)
    return rmse, iae, itae


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynTrialResult:
    trial_idx:          int
    seed:               int
    n_dynamic_obs:      int
    goal_dist_init:     float
    success:            bool
    collision:          bool
    timeout:            bool
    dynamic_obs_hit:    bool
    final_dist:         float
    min_dist:           float
    convergence_step:   int
    steps_taken:        int
    path_length:        float
    mean_dist:          float
    rmse:               float
    iae:                float
    itae:               float
    plr:                float
    fz_mean:            float
    fz_drift:           float
    z_drift:            float
    z_max:              float
    min_obs_clearance:  float
    near_miss_count:    int
    first_appear_step:  int
    reaction_step:      int
    wall_time_s:        float


@dataclass
class BenchmarkStats:
    n_trials:               int
    n_dynamic_obs:          int
    success_rate:           float
    collision_rate:         float
    timeout_rate:           float
    dynamic_hit_rate:       float
    mean_final_dist:        float
    std_final_dist:         float
    mean_min_dist:          float
    std_min_dist:           float
    mean_convergence_step:  float
    mean_path_length:       float
    std_path_length:        float
    mean_rmse:              float
    std_rmse:               float
    mean_iae:               float
    std_iae:                float
    mean_itae:              float
    std_itae:               float
    mean_plr:               float
    std_plr:                float
    mean_fz_drift:          float
    std_fz_drift:           float
    mean_z_drift:           float
    std_z_drift:            float
    mean_obs_clearance:     float
    std_obs_clearance:      float
    mean_near_miss_count:   float
    std_near_miss_count:    float
    mean_reaction_step:     float
    std_reaction_step:      float
    mean_wall_time_s:       float


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _rk4_step(dyn, t, x, u, dt=DT):
    def f(tt, xx): return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t,       x)
    k2 = f(t+.5*dt, x+.5*dt*k1)
    k3 = f(t+.5*dt, x+.5*dt*k2)
    k4 = f(t+dt,    x+dt*k3)
    xn = x + (dt/6.)*(k1+2*k2+2*k3+k4)
    q  = xn[6:10]; xn[6:10] = q/(np.linalg.norm(q)+1e-12)
    return xn


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE TRIAL
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(trial_idx: int, seed: int, n_obs: int,
              verbose: bool = False,
              save_dir: Optional[str] = None) -> Tuple[DynTrialResult, dict]:

    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    params   = dynamics.QuadrotorParams()
    dyn      = dynamics.QuadrotorDynamics(params)

    ctrl     = UHRCWaypointController(MODEL_PATH, STATS_PATH, device="cpu")
    ctrl.reset()

    att_ctrl = _make_att_ctrl(params)
    att_ctrl.reset()

    start, goal = _sample_start_goal(rng)
    pop_obs     = build_pop_obstacles(n_obs, start, goal, rng)

    goal_dist_init = float(np.linalg.norm(goal[:2]-start[:2]))
    first_appear   = min((o.appear_step for o in pop_obs), default=-1)

    x_curr = dyn.pack_state(
        start, np.zeros(3), np.array([1., 0., 0., 0.]),
        np.zeros(3), np.zeros(4))

    # ── Buffers ───────────────────────────────────────────────────────────────
    positions     = [start.copy()]
    z_log         = [float(start[2])]
    dists         = []
    fz_log        = []
    tau_x_log     = []
    tau_y_log     = []
    tau_z_log     = []
    phi_ref_log   = []
    theta_ref_log = []
    subgoal_log   = []
    obs_clear_log = []
    heading_log   = []
    near_miss_count = 0
    reaction_step   = -1
    appeared_flag   = False

    t_sim      = 0.0
    success    = False
    collision  = False
    dyn_hit    = False
    conv_step  = -1

    # Per-step dynamic positions for saving [T, n_obs, 2]
    dyn_pos_log: List[List] = []

    for step in range(MAX_STEPS):
        r_I, v_I, q_BI, w_B, Omega = dyn.unpack_state(x_curr)
        psi = float(quat_euler.euler_from_q(q_BI)[2])
        heading_log.append(psi)

        # Active obstacle list for lidar + collision
        active_circles = [o.to_circle_tuple() for o in pop_obs if o.is_active(step)]
        dyn_pos_log.append([(o.cx, o.cy) for o in pop_obs])

        # ── Lidar sees only active obstacles ──────────────────────────────────
        lidar = get_lidar_scan(r_I, psi, active_circles, num_rays=32,
                               fov=math.pi, max_range=LIDAR_MAX)

        # ── Reaction detection ────────────────────────────────────────────────
        just_appeared = any(o.appear_step == step for o in pop_obs)
        if just_appeared and not appeared_flag:
            appeared_flag = True
        if appeared_flag and reaction_step == -1 and len(heading_log) >= 2:
            dh = abs(heading_log[-1] - heading_log[-2])
            dh = min(dh, 2*math.pi - dh)
            if dh > 0.02:
                reaction_step = step

        # ── Outer loop: UHRC → [phi_ref, theta_ref] ───────────────────────────
        att_ref, sub_nn = ctrl.get_att_ref(r_I, v_I, q_BI, w_B,
                                           Omega, lidar, goal)
        phi_ref   = float(np.clip(att_ref[0], -MAX_TILT_RAD, MAX_TILT_RAD))
        theta_ref = float(np.clip(att_ref[1], -MAX_TILT_RAD, MAX_TILT_RAD))

        # ── Inner loop: AttitudePID → [Fz, τx, τy, τz] ───────────────────────
        refs = {'phi': phi_ref, 'theta': theta_ref, 'psi': 0.0, 'z': HOVER_Z}
        u    = att_ctrl.step(x_curr, refs, DT)

        fz_log.append(float(u[0]))
        tau_x_log.append(float(u[1]))
        tau_y_log.append(float(u[2]))
        tau_z_log.append(float(u[3]))
        phi_ref_log.append(phi_ref)
        theta_ref_log.append(theta_ref)
        subgoal_log.append(sub_nn.copy() if sub_nn is not None else np.zeros(2))

        dist = float(np.linalg.norm(r_I[:2]-goal[:2]))
        dists.append(dist)
        if conv_step == -1 and dist < CONVERGE_THRESH:
            conv_step = step

        # Clearances — only against active obstacles
        clearances = [
            float(np.linalg.norm(r_I[:2]-np.asarray(c[:2]))) - float(r_obs)
            for c, r_obs in active_circles
        ]
        if clearances:
            min_cl = min(clearances)
            obs_clear_log.append(min_cl)
            if min_cl < NEAR_MISS_THRESH:
                near_miss_count += 1

        if verbose and step % 100 == 0:
            n_active = len(active_circles)
            print(f"  trial {trial_idx} step {step:4d}"
                  f"  pos=({r_I[0]:6.2f},{r_I[1]:6.2f})"
                  f"  z={r_I[2]:+.3f}  dist={dist:.2f}m"
                  f"  active_obs={n_active}"
                  f"  Fz={u[0]:.2f}")

        x_curr  = _rk4_step(dyn, t_sim, x_curr, u)
        t_sim  += DT
        r_new, *_ = dyn.unpack_state(x_curr)
        positions.append(r_new.copy())
        z_log.append(float(r_new[2]))

        # Collision checks — only active obstacles
        next_active = [o.to_circle_tuple() for o in pop_obs
                       if o.is_active(step+1)]
        hit_dyn = any(
            float(np.linalg.norm(r_new[:2]-np.asarray(c[:2]))) < float(r_obs)
            for c, r_obs in next_active)
        reached = float(np.linalg.norm(r_new[:2]-goal[:2])) < GOAL_RADIUS

        if hit_dyn:
            collision = True; dyn_hit = True; break
        if reached:
            success = True; break

    # ── Metrics ───────────────────────────────────────────────────────────────
    steps_taken  = len(positions) - 1
    path         = np.array(positions)
    path_length  = float(np.sum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)))
    final_dist   = float(np.linalg.norm(path[-1,:2]-goal[:2]))
    min_dist     = float(min(dists)) if dists else goal_dist_init
    mean_dist    = float(np.mean(dists)) if dists else goal_dist_init

    rmse, iae, itae = _compute_integral_metrics(dists, DT)
    plr  = path_length / goal_dist_init if goal_dist_init > 1e-3 else 1.0

    fz_arr   = np.array(fz_log) if fz_log else np.full(1, params.mass*params.g)
    fz_mean  = float(fz_arr.mean())
    fz_drift = float(np.mean(np.abs(fz_arr - params.mass*params.g)))

    z_arr    = np.array(z_log)
    z_drift  = float(np.mean(np.abs(z_arr - HOVER_Z)))
    z_max    = float(np.max(np.abs(z_arr - HOVER_Z)))

    min_obs_clearance = float(min(obs_clear_log)) if obs_clear_log else -1.0
    wall_time = time.perf_counter() - t0

    result = DynTrialResult(
        trial_idx=trial_idx, seed=seed,
        n_dynamic_obs=len(pop_obs), goal_dist_init=goal_dist_init,
        success=success, collision=collision,
        timeout=not success and not collision,
        dynamic_obs_hit=dyn_hit,
        final_dist=final_dist, min_dist=min_dist,
        convergence_step=conv_step, steps_taken=steps_taken,
        path_length=path_length, mean_dist=mean_dist,
        rmse=rmse, iae=iae, itae=itae, plr=plr,
        fz_mean=fz_mean, fz_drift=fz_drift,
        z_drift=z_drift, z_max=z_max,
        min_obs_clearance=min_obs_clearance,
        near_miss_count=near_miss_count,
        first_appear_step=first_appear,
        reaction_step=reaction_step,
        wall_time_s=wall_time,
    )

    # ── Serialise ─────────────────────────────────────────────────────────────
    dyn_arr = np.array([[o.cx, o.cy, o.r, o.appear_step]
                        for o in pop_obs], dtype=np.float32)

    max_len = min(len(dyn_pos_log), steps_taken)
    dyn_traj = (np.array([[list(p) for p in frame]
                           for frame in dyn_pos_log[:max_len]], dtype=np.float32)
                if pop_obs and max_len > 0
                else np.zeros((max_len, 0, 2), dtype=np.float32))

    trial_data = {
        "trajectory":       path.astype(np.float32),
        "start":            start.astype(np.float32),
        "goal":             goal.astype(np.float32),
        "dynamic_obs":      dyn_arr,          # [n_obs, 4]: cx cy r appear_step
        "dynamic_traj":     dyn_traj,         # [T, n_obs, 2]: positions over time
        "success":          np.array([success],              dtype=bool),
        "collision":        np.array([collision],            dtype=bool),
        "timeout":          np.array([not success and not collision], dtype=bool),
        "dynamic_obs_hit":  np.array([dyn_hit],              dtype=bool),
        "final_dist":       np.array([final_dist],           dtype=np.float32),
        "rmse":             np.array([rmse],                 dtype=np.float32),
        "iae":              np.array([iae],                  dtype=np.float32),
        "itae":             np.array([itae],                 dtype=np.float32),
        "plr":              np.array([plr],                  dtype=np.float32),
        "seed":             np.array([seed],                 dtype=np.int64),
        "dist_log":         np.array(dists,         dtype=np.float32),
        "fz_log":           fz_arr.astype(np.float32),
        "tau_x_log":        np.array(tau_x_log,     dtype=np.float32),
        "tau_y_log":        np.array(tau_y_log,     dtype=np.float32),
        "tau_z_log":        np.array(tau_z_log,     dtype=np.float32),
        "phi_ref_log":      np.array(phi_ref_log,   dtype=np.float32),
        "theta_ref_log":    np.array(theta_ref_log, dtype=np.float32),
        "z_log":            z_arr.astype(np.float32),
        "subgoal_log":      np.array(subgoal_log,   dtype=np.float32),
        "near_miss_count":  np.array([near_miss_count], dtype=np.int32),
        "reaction_step":    np.array([reaction_step],    dtype=np.int32),
        "first_appear_step":np.array([first_appear],     dtype=np.int32),
    }

    if save_dir is not None:
        out_path = os.path.join(save_dir, "trials",
                                f"dyn_trial{trial_idx:03d}.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **trial_data)

    return result, trial_data


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(trials: List[DynTrialResult]) -> BenchmarkStats:
    def _m(arr): return float(np.mean(arr)) if arr else float("nan")
    def _s(arr): return float(np.std(arr))  if arr else float("nan")

    fd   = [t.final_dist        for t in trials]
    md   = [t.min_dist          for t in trials]
    pl   = [t.path_length       for t in trials]
    rm   = [t.rmse              for t in trials]
    ia   = [t.iae               for t in trials]
    ita  = [t.itae              for t in trials]
    plr  = [t.plr               for t in trials]
    fzd  = [t.fz_drift          for t in trials]
    zd   = [t.z_drift           for t in trials]
    wt   = [t.wall_time_s       for t in trials]
    clr  = [t.min_obs_clearance for t in trials if t.min_obs_clearance >= 0]
    nm   = [t.near_miss_count   for t in trials]
    cvk  = [t.convergence_step  for t in trials if t.convergence_step  >= 0]
    rct  = [t.reaction_step     for t in trials if t.reaction_step     >= 0]

    return BenchmarkStats(
        n_trials=len(trials),
        n_dynamic_obs=trials[0].n_dynamic_obs,
        success_rate      =_m([t.success        for t in trials]),
        collision_rate    =_m([t.collision       for t in trials]),
        timeout_rate      =_m([t.timeout         for t in trials]),
        dynamic_hit_rate  =_m([t.dynamic_obs_hit for t in trials]),
        mean_final_dist=_m(fd), std_final_dist=_s(fd),
        mean_min_dist  =_m(md), std_min_dist  =_s(md),
        mean_convergence_step=_m(cvk),
        mean_path_length=_m(pl), std_path_length=_s(pl),
        mean_rmse=_m(rm),  std_rmse=_s(rm),
        mean_iae =_m(ia),  std_iae =_s(ia),
        mean_itae=_m(ita), std_itae=_s(ita),
        mean_plr =_m(plr), std_plr =_s(plr),
        mean_fz_drift=_m(fzd), std_fz_drift=_s(fzd),
        mean_z_drift =_m(zd),  std_z_drift =_s(zd),
        mean_obs_clearance  =_m(clr), std_obs_clearance  =_s(clr),
        mean_near_miss_count=_m(nm),  std_near_miss_count=_s(nm),
        mean_reaction_step  =_m(rct), std_reaction_step  =_s(rct),
        mean_wall_time_s=_m(wt),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(stats: BenchmarkStats, n_obs: int):
    sep = "─" * 70
    print("\n" + sep)
    print(f"  {TC_ID} {SCENARIO_NAME}  |  {stats.n_trials} trials"
          f"  |  {n_obs} dynamic pop-in obstacles per episode")
    print(sep)

    # Outcomes
    print(f"  Outcomes:")
    print(f"    Success      = {stats.success_rate*100:.1f}%")
    print(f"    Collision    = {stats.collision_rate*100:.1f}%"
          f"  (dynamic-obs hit = {stats.dynamic_hit_rate*100:.1f}%)")
    print(f"    Timeout      = {stats.timeout_rate*100:.1f}%")

    # Tracking metrics
    print(f"\n  Tracking (RMSE / IAE / ITAE / PLR):")
    print(f"    RMSE   = {stats.mean_rmse:.3f} ± {stats.std_rmse:.3f} m")
    print(f"    IAE    = {stats.mean_iae:.3f} ± {stats.std_iae:.3f} m·s")
    print(f"    ITAE   = {stats.mean_itae:.3f} ± {stats.std_itae:.3f} m·s²")
    print(f"    PLR    = {stats.mean_plr:.3f} ± {stats.std_plr:.3f}")

    # Navigation
    print(f"\n  Navigation:")
    print(f"    FinalDist    = {stats.mean_final_dist:.3f} ± {stats.std_final_dist:.3f} m")
    print(f"    PathLen      = {stats.mean_path_length:.3f} ± {stats.std_path_length:.3f} m")
    rct_str = (f"{stats.mean_reaction_step:.0f} ± {stats.std_reaction_step:.0f} steps"
               if not math.isnan(stats.mean_reaction_step) else "n/a")
    print(f"    ReactionStep = {rct_str}")
    print(f"    NearMiss     = {stats.mean_near_miss_count:.1f} ± "
          f"{stats.std_near_miss_count:.1f} steps/ep  "
          f"(< {NEAR_MISS_THRESH} m clearance)")

    # Control quality
    print(f"\n  Control quality:")
    print(f"    Fz drift     = {stats.mean_fz_drift:.3f} ± {stats.std_fz_drift:.3f} N")
    print(f"    z drift      = {stats.mean_z_drift:.3f} ± {stats.std_z_drift:.3f} m")
    if not math.isnan(stats.mean_obs_clearance):
        print(f"    Min clearance= {stats.mean_obs_clearance:.3f} ± "
              f"{stats.std_obs_clearance:.3f} m")
    print(sep)


def save_results(trials: List[DynTrialResult], stats: BenchmarkStats,
                 out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    td = [asdict(t) for t in trials]
    if td:
        with open(os.path.join(out_dir, "trials_dynamic.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=td[0].keys())
            w.writeheader(); w.writerows(td)
    sd = asdict(stats)
    with open(os.path.join(out_dir, "summary_dynamic.json"), "w") as f:
        json.dump({"summary": sd, "trials": td}, f, indent=2)
    print(f"  Results saved → {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(stats: BenchmarkStats, all_trials: List[DynTrialResult],
                 out_dir: str):
    fig, axes = plt.subplots(2, 4, figsize=(26, 10))
    fig.suptitle(
        f"UHRC Dynamic Benchmark ({TC_ID}) — "
        f"UHRC(att-ref) + AttitudePID(inner)",
        fontsize=13, y=0.98)

    n  = stats.n_trials
    bw = 0.55

    # Outcome bar
    ax = axes[0, 0]
    cats   = ["Success", "Collision", "Timeout"]
    values = [stats.success_rate*100,
              stats.collision_rate*100,
              stats.timeout_rate*100]
    colors = ["#4CAF50", "#F44336", "#FFC107"]
    ax.bar(range(3), values, bw, color=colors)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.set_title("Outcome")
    for i, v in enumerate(values):
        ax.text(i, v+1, f"{v:.1f}%", ha="center", fontsize=8)

    # Final distance histogram
    ax = axes[0, 1]
    fd = [t.final_dist for t in all_trials]
    ax.hist(fd, bins=15, color="#2196F3", alpha=0.85, edgecolor="white")
    ax.axvline(GOAL_RADIUS, color="#4CAF50", ls="--", lw=1.5,
               label=f"Goal {GOAL_RADIUS}m")
    ax.set_xlabel("Final dist (m)"); ax.set_ylabel("Count")
    ax.set_title(f"Final distance  mean={stats.mean_final_dist:.2f}m")
    ax.legend(fontsize=7)

    # Near-miss count histogram
    ax = axes[0, 2]
    nm = [t.near_miss_count for t in all_trials]
    ax.hist(nm, bins=max(10, max(nm, default=1)+1), color="#FF5722",
            alpha=0.85, edgecolor="white")
    ax.set_xlabel("Near-miss steps"); ax.set_ylabel("Count")
    ax.set_title(f"Near-miss count  (clearance < {NEAR_MISS_THRESH}m)\n"
                 f"mean={stats.mean_near_miss_count:.1f}")

    # Reaction step histogram
    ax = axes[0, 3]
    rct = [t.reaction_step for t in all_trials if t.reaction_step >= 0]
    if rct:
        ax.hist(rct, bins=15, color="#607D8B", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Step"); ax.set_ylabel("Count")
        rct_str = (f"{stats.mean_reaction_step:.0f}±{stats.std_reaction_step:.0f}"
                   if not math.isnan(stats.mean_reaction_step) else "n/a")
        ax.set_title(f"Reaction step  mean={rct_str}")
    else:
        ax.text(0.5, 0.5, "No reaction detected", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Reaction step")

    # RMSE histogram
    ax = axes[1, 0]
    rm = [t.rmse for t in all_trials]
    ax.hist(rm, bins=15, color="#9C27B0", alpha=0.85, edgecolor="white")
    ax.set_xlabel("RMSE (m)"); ax.set_ylabel("Count")
    ax.set_title(f"RMSE  mean={stats.mean_rmse:.3f}m")

    # Fz drift histogram
    ax = axes[1, 1]
    fzd = [t.fz_drift for t in all_trials]
    ax.hist(fzd, bins=15, color="#FF9800", alpha=0.85, edgecolor="white")
    ax.axvline(0.5, color="#F44336", ls="--", lw=1.2, label="0.5N")
    ax.set_xlabel("|Fz − mg| (N)"); ax.set_ylabel("Count")
    ax.set_title(f"Fz drift  mean={stats.mean_fz_drift:.3f}N")
    ax.legend(fontsize=7)

    # Altitude drift histogram
    ax = axes[1, 2]
    zd = [t.z_drift for t in all_trials]
    ax.hist(zd, bins=15, color="#E91E63", alpha=0.85, edgecolor="white")
    ax.axvline(0.2, color="#F44336", ls="--", lw=1.2, label="0.2m")
    ax.set_xlabel("|z − 1m| (m)"); ax.set_ylabel("Count")
    ax.set_title(f"Altitude drift  mean={stats.mean_z_drift:.3f}m")
    ax.legend(fontsize=7)

    # PLR histogram
    ax = axes[1, 3]
    plr = [t.plr for t in all_trials]
    ax.hist(plr, bins=15, color="#00BCD4", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="#4CAF50", ls="--", lw=1.2, label="PLR=1 (straight)")
    ax.set_xlabel("PLR"); ax.set_ylabel("Count")
    ax.set_title(f"Path Length Ratio  mean={stats.mean_plr:.3f}")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "benchmark_summary_dynamic.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


def plot_trial(trial_data: dict, result: DynTrialResult, out_path: str):
    """Plot XY trajectory with dynamic obstacle positions overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    traj     = trial_data["trajectory"]
    dyn_arr  = trial_data["dynamic_obs"]    # [n_obs, 4]: cx cy r appear_step
    dyn_traj = trial_data.get("dynamic_traj")  # [T, n_obs, 2]

    # ── Left: XY trajectory ───────────────────────────────────────────────
    ax = axes[0]

    for i in range(len(dyn_arr)):
        cx0, cy0, r, appear = dyn_arr[i]
        ax.add_patch(Circle((cx0, cy0), r, ls="--", lw=1.5,
                            edgecolor="#9C27B0", facecolor="#CE93D8",
                            alpha=0.35, zorder=2))
        ax.text(cx0, cy0, f"D{i}\n@s{int(appear)}",
                ha="center", va="center", fontsize=6, color="#6A1B9A", zorder=4)
        if dyn_traj is not None and dyn_traj.shape[1] > i:
            trail = dyn_traj[:, i, :]
            appear_step = int(appear)
            if appear_step < len(trail):
                ax.plot(trail[appear_step:, 0], trail[appear_step:, 1],
                        color="#9C27B0", lw=1.0, ls=":", alpha=0.55, zorder=3)
                if len(trail) > appear_step + 1:
                    fx, fy = trail[-1]
                    ax.add_patch(Circle((fx, fy), r, lw=1.2,
                                        edgecolor="#9C27B0", facecolor="#E1BEE7",
                                        alpha=0.50, zorder=3))

    clr   = ("royalblue" if result.success
              else "red" if result.collision else "orange")
    label = ("Success" if result.success
             else "Collision (dyn)" if result.dynamic_obs_hit
             else "Collision" if result.collision else "Timeout")
    ax.plot(traj[:, 0], traj[:, 1], color=clr, lw=2, zorder=5,
            label=f"UHRC ({label})")
    ax.plot(*trial_data["start"][:2], "go", ms=10, label="Start", zorder=6)
    ax.plot(*trial_data["goal"][:2],  "b*", ms=14, label="Goal",  zorder=6)
    ax.add_patch(Circle(trial_data["goal"][:2], GOAL_RADIUS,
                        fill=False, ls="--", color="blue", alpha=0.4))

    ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(ARENA); ax.set_ylim(ARENA)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{TC_ID} {SCENARIO_NAME}  seed={result.seed}"
                 f"  RMSE={result.rmse:.3f}m  PLR={result.plr:.2f}")

    # ── Right: distance-to-goal + near-miss markers ───────────────────────
    ax2 = axes[1]
    dist_log = trial_data["dist_log"]
    t_ax     = np.arange(len(dist_log)) * DT
    ax2.plot(t_ax, dist_log, color="royalblue", lw=1.6, label="Dist to goal")
    ax2.axhline(GOAL_RADIUS, color="green", ls="--", lw=1.0,
                label=f"Goal {GOAL_RADIUS}m")
    ax2.axhline(NEAR_MISS_THRESH, color="orange", ls=":", lw=1.0,
                label=f"Near-miss {NEAR_MISS_THRESH}m")

    for i in range(len(dyn_arr)):
        appear_step = int(dyn_arr[i, 3])
        appear_t    = appear_step * DT
        if appear_t < t_ax[-1]:
            ax2.axvline(appear_t, color="#9C27B0", ls="--", lw=0.9, alpha=0.7,
                        label=f"D{i} appears" if i == 0 else "")
            ax2.text(appear_t,
                     float(dist_log[min(appear_step, len(dist_log)-1)]),
                     f" D{i}", fontsize=7, color="#6A1B9A")

    if result.reaction_step >= 0:
        rt = result.reaction_step * DT
        ax2.axvline(rt, color="red", ls="-.", lw=0.9, alpha=0.7,
                    label=f"Reaction@s{result.reaction_step}")

    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Distance (m)")
    ax2.set_title(f"dist={result.final_dist:.2f}m  "
                  f"IAE={result.iae:.2f}  ITAE={result.itae:.1f}  "
                  f"nm={result.near_miss_count}  "
                  f"z_drift={result.z_drift:.3f}m")
    ax2.legend(fontsize=7); ax2.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="UHRC dynamic random-pop obstacle benchmark")
    p.add_argument("--n",         type=int, default=DEFAULT_TRIALS,
                   help="Number of trials")
    p.add_argument("--n_obs",     type=int, default=DEFAULT_N_OBS,
                   help="Number of pop-in obstacles per episode")
    p.add_argument("--seed_base", type=int, default=1000)
    p.add_argument("--out",       type=str, default="benchmark_results_dynamic")
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--quick",     action="store_true",
                   help="Run 5 trials for a quick smoke-test")
    p.add_argument("--plot_trials", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    n_trials = 5 if args.quick else args.n
    n_obs    = args.n_obs

    print(f"\n{'='*60}")
    print(f"  UHRC Dynamic Benchmark — {TC_ID} {SCENARIO_NAME}")
    print(f"  Outer : UHRCWaypointController.get_att_ref() → [φ_ref, θ_ref]")
    print(f"  Inner : AttitudePID.step()  → [Fz, τx, τy, τz]")
    print(f"  Tilt  : ±{math.degrees(MAX_TILT_RAD):.0f}°  |  HOVER_Z={HOVER_Z}m")
    print(f"  Pop-in obstacles per episode : {n_obs}")
    print(f"  Pop window : {POP_WINDOW_FRAC[0]*100:.0f}–{POP_WINDOW_FRAC[1]*100:.0f}%"
          f" of MAX_STEPS ({MAX_STEPS})")
    print(f"  Near-miss threshold          : {NEAR_MISS_THRESH}m")
    print(f"  Trials : {n_trials}  |  Model : {MODEL_PATH}")
    print(f"  Output : {args.out}/")
    print(f"{'='*60}\n")

    all_trials: List[DynTrialResult] = []

    for i in range(n_trials):
        seed = args.seed_base + i
        tr, td = run_trial(i, seed, n_obs,
                           verbose=args.verbose,
                           save_dir=args.out)
        all_trials.append(tr)

        icon = "✅" if tr.success else ("💥D" if tr.dynamic_obs_hit
                                        else "💥" if tr.collision else "⏱")
        rct  = (f"react@{tr.reaction_step}"
                if tr.reaction_step >= 0 else "no-react")
        print(f"  [{i+1:4d}/{n_trials}]  {icon}"
              f"  dist={tr.final_dist:5.2f}m"
              f"  RMSE={tr.rmse:.3f}m"
              f"  IAE={tr.iae:.2f}"
              f"  PLR={tr.plr:.2f}"
              f"  nm={tr.near_miss_count:3d}"
              f"  {rct}"
              f"  z_drift={tr.z_drift:.3f}m", flush=True)

        if args.plot_trials:
            fig_dir = os.path.join(args.out, "plots")
            os.makedirs(fig_dir, exist_ok=True)
            plot_trial(td, tr,
                       os.path.join(fig_dir, f"dyn_trial{i:03d}.png"))

    stats = compute_stats(all_trials)
    print_summary(stats, n_obs)
    save_results(all_trials, stats, args.out)
    plot_summary(stats, all_trials, args.out)

    total_succ = sum(t.success for t in all_trials)
    print(f"\n  Overall: {total_succ}/{n_trials}"
          f"  ({total_succ/n_trials*100:.1f}% success)\n")


if __name__ == "__main__":
    main()