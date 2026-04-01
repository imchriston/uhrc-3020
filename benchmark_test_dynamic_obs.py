"""
eval_uhrc_benchmark.py
Structured Monte-Carlo benchmark for the UHRC controller.

Aligned with the Comparative Test Plan (ECNG 3020).
Logs ALL metrics required for:
  - Tracking & performance comparison (RMSE, IAE, ISE, ITAE)



Usage
-----
  python eval_uhrc_benchmark.py                          # full benchmark
  python eval_uhrc_benchmark.py --scenario dynamic_obstacles --n 30 
  Add --save_timeseries to save each trial

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import functools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import drone.dynamics as dynamics
from generate_data import get_lidar_scan
from uhrc_ctrl import UHRCController
import utils.quat_euler as quat_euler

# ── Global config ──────────────────────────────────────────────────────────────
MODEL_PATH       = "checkpoints/uhrc_best.pth"
STATS_PATH       = "checkpoints/norm_stats.npz"
DT               = 0.01
MAX_STEPS        = 1500
GOAL_RADIUS      = 0.5
CONVERGE_THRESH  = 1.0
HOVER_THRUST     = 9.81
ARENA            = (-10.0, 10.0)
DEFAULT_TRIALS   = 30
NEAR_MISS_THRESH = 0.3
LYAPUNOV_ALPHA   = 0.1

DYN_OBS_APPEAR_TIME = 3.0
DYN_OBS_POSITIONS   = [(2.0, 0.5, 0.0), (3.0, -0.5, 0.0)]
DYN_OBS_RADIUS      = 0.8
DYN_OBS_DETECT_DIST = 5.0   # sensor range at which dynamic obs counts as "detected"
DYN_OBS_MAX_STEPS   = int(15.0 / DT)   # 15 s cap per spec


OBS_COUNT = {
    "open_field":         0,
    "dense":              6,
    "narrow_corridor":    4,
    "gap_navigation":     4,
    "close_range":        2,
    "long_range":         4,
    "dynamic_obstacles":  4,   
    "urban_corridor":     6,   
    "urban_dense":        10,  
}


def _patch_controller_for_preclamp(ctrl: UHRCController):
    original_infer = ctrl._infer

    @functools.wraps(original_infer)
    def patched_infer(obs, goal_dist=999.0):
        import torch
        x_norm  = (torch.from_numpy(obs).float().to(ctrl.device) - ctrl.obs_mean) / ctrl.obs_std
        x_input = x_norm.unsqueeze(0)
        with torch.no_grad():
            action_t, subgoal_t, ctrl.carry = ctrl.model(x_input, carry=ctrl.carry)
        action  = action_t[0].cpu().numpy()
        subgoal = subgoal_t[0].cpu().numpy()
        ctrl._last_raw_action = action.copy()
        if ctrl.carry is not None:
            zh_norm = float(torch.norm(ctrl.carry.z_H).item())
            zl_norm = float(torch.norm(ctrl.carry.z_L).item())
        else:
            zh_norm = zl_norm = 0.0
        ctrl._last_carry_norms = (zh_norm, zl_norm)
        action[0] = float(np.clip(action[0],  4.0, 20.0))
        action[1] = float(np.clip(action[1], -0.3,  0.3))
        action[2] = float(np.clip(action[2], -0.3,  0.3))
        action[3] = float(np.clip(action[3], -0.3,  0.3))
        return action, subgoal

    ctrl._infer            = patched_infer
    ctrl._last_raw_action  = np.zeros(4)
    ctrl._last_carry_norms = (0.0, 0.0)


@dataclass
class TrialResult:
    scenario:          str
    trial_idx:         int
    seed:              int
    n_obstacles:       int
    goal_dist_init:    float
    # Outcomes
    success:           bool
    collision:         bool
    timeout:           bool
    # Distance
    final_dist:        float
    min_dist:          float
    mean_dist:         float
    # Integral error
    rmse:              float
    iae:               float
    ise:               float
    itae:              float
    # Navigation
    path_length:       float
    path_length_ratio: float
    near_miss_count:   int
    completion_time_s: float
    # Control quality
    convergence_step:    int
    settling_time_s:     float
    overshoot:           float
    steady_state_error:  float
    action_smoothness:   float
    subgoal_consistency: float
    # Thrust
    fz_mean:   float
    fz_max:    float
    fz_drift:  float
    steps_taken: int
    # Obstacle clearance
    min_obs_clearance: float
    # Stability
    lyapunov_v_final:       float
    lyapunov_violations:    int
    lyapunov_max_dv:        float
    carry_norm_zh_max:      float
    carry_norm_zl_max:      float
    carry_norm_zh_final:    float
    carry_norm_zl_final:    float
    clamp_activations:      int
    max_consecutive_clamps: int
    # Per-axis
    rmse_x: float
    rmse_y: float
    # TC-10: dynamic obstacle metrics
    # replanning_latency_s: seconds from first detection to first avoidance subgoal
    #   change; -1 if drone never detected them (passed before t=3s)
    replanning_latency_s:    float
    dyn_obs_near_miss_count: int
    dyn_obs_min_clearance:   float
    itae_post_event:         float   # ITAE for steps at t >= DYN_OBS_APPEAR_TIME
    # TC-11: actuator fault metrics
    fault_severity:       float   # motor effectiveness (1.0 = no fault)
    max_roll_deg:         float
    max_pitch_deg:        float
    rmse_nominal_delta:   float   # RMSE - nominal_RMSE; -1 if not a fault run
    # Timing
    wall_time_s: float


@dataclass
class ScenarioStats:
    scenario:             str
    n_trials:             int
    n_obstacles:          int
    success_rate:         float
    collision_rate:       float
    timeout_rate:         float
    mean_rmse:            float
    std_rmse:             float
    mean_iae:             float
    mean_ise:             float
    mean_itae:            float
    mean_path_length:     float
    std_path_length:      float
    mean_plr:             float
    mean_near_miss:       float
    mean_completion_time: float
    mean_settling_time:   float
    mean_overshoot:       float
    mean_ss_error:        float
    mean_action_smooth:   float
    mean_subgoal_consist: float
    mean_final_dist:      float
    std_final_dist:       float
    mean_min_dist:        float
    mean_lyap_violations: float
    mean_carry_zh_max:    float
    mean_carry_zl_max:    float
    mean_clamp_acts:      float
    mean_fz_drift:        float
    std_fz_drift:         float
    mean_obs_clearance:   float
    std_obs_clearance:    float
    mean_wall_time_s:     float
    # TC-10 (nan when not dynamic scenario)
    mean_replanning_latency_s:  float
    mean_dyn_obs_near_miss:     float
    mean_dyn_obs_min_clearance: float
    mean_itae_post_event:       float
    # TC-11 (nan when not fault scenario)
    fault_severity:           float
    mean_max_roll_deg:        float
    mean_max_pitch_deg:       float
    mean_rmse_nominal_delta:  float


# ── RK4 ───────────────────────────────────────────────────────────────────────
def _rk4_step(dyn, t, x, u, dt=DT, fault_mask=None):
    u_eff = np.array(u, dtype=float)
    if fault_mask is not None:
        u_eff[0] *= np.mean(fault_mask)
        u_eff[1] *= np.min(fault_mask)
        u_eff[2] *= np.min(fault_mask)
        u_eff[3] *= np.mean(fault_mask)

    def f(tt, xx):
        return dyn.f(tt, xx, u_eff, "body_wrench")
    k1 = f(t,         x)
    k2 = f(t + .5*dt, x + .5*dt*k1)
    k3 = f(t + .5*dt, x + .5*dt*k2)
    k4 = f(t +    dt, x +    dt*k3)
    xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q = xn[6:10]
    xn[6:10] = q / (np.linalg.norm(q) + 1e-12)
    return xn


# ── Obstacle generators ───────────────────────────────────────────────────────
def _min_sep(centers, candidate, radii, r_cand, margin=0.3):
    for c, r in zip(centers, radii):
        if np.linalg.norm(np.array(c) - np.array(candidate)) < r + r_cand + margin:
            return False
    return True

def _safe_obs(cx, cy, r, start, goal, min_path_gap=0.8):
    for pt in (start[:2], goal[:2]):
        if np.linalg.norm(np.array([cx, cy]) - np.array(pt)) < r + min_path_gap:
            return False
    return True

def build_obstacles_random(n, start, goal, rng):
    obs, centers, radii = [], [], []
    attempts = 0
    while len(obs) < n and attempts < 500:
        cx = float(rng.uniform(*ARENA))
        cy = float(rng.uniform(*ARENA))
        r  = float(rng.uniform(0.5, 1.2))
        attempts += 1
        if not _safe_obs(cx, cy, r, start, goal):
            continue
        if not _min_sep(centers, (cx, cy), radii, r):
            continue
        obs.append(((cx, cy, 0.0), r))
        centers.append((cx, cy))
        radii.append(r)
    return obs

def build_obstacles_narrow_corridor(start, goal, rng):
    mid  = (start[:2] + goal[:2]) / 2.0
    perp = np.array([-(goal[1]-start[1]), goal[0]-start[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    r    = 0.7
    gh   = 0.8 + float(rng.uniform(0.0, 0.4))
    foff = float(rng.uniform(-1.5, 1.5))
    fwd  = goal[:2] - start[:2]; fwd /= np.linalg.norm(fwd) + 1e-9
    obs  = []
    for c in (mid+perp*(gh+r)+fwd*foff, mid-perp*(gh+r)+fwd*foff,
              mid+perp*(gh+r)-fwd*foff, mid-perp*(gh+r)-fwd*foff):
        if ARENA[0]+1 < c[0] < ARENA[1]-1 and ARENA[0]+1 < c[1] < ARENA[1]-1:
            obs.append(((float(c[0]), float(c[1]), 0.0), r))
    while len(obs) < 4:
        obs += build_obstacles_random(1, start, goal, rng)
    return obs[:4]

def build_obstacles_gap(start, goal, rng):
    mid  = (start[:2] + goal[:2]) / 2.0
    perp = np.array([-(goal[1]-start[1]), goal[0]-start[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    rl   = 0.9
    gp   = float(rng.uniform(-0.5, 0.5))
    fwd  = goal[:2] - start[:2]; fwd /= np.linalg.norm(fwd) + 1e-9
    obs  = []
    for c in (mid+perp*(gp+2.1*rl), mid+perp*(gp-2.1*rl),
              mid+perp*(gp+5.0*rl), mid+fwd*2.0+perp*float(rng.uniform(-1.5,1.5))):
        if ARENA[0]+1 < c[0] < ARENA[1]-1 and ARENA[0]+1 < c[1] < ARENA[1]-1:
            obs.append(((float(c[0]), float(c[1]), 0.0), rl))
    while len(obs) < 4:
        obs += build_obstacles_random(1, start, goal, rng)
    return obs[:4]

def _rect_to_circle_list(buildings):
    """Convert rect building dicts to the ((cx,cy,0), r) circle format used
    everywhere else, using the circumradius as a conservative collision proxy."""
    return [((b["cx"], b["cy"], 0.0),
             float(np.hypot(b["sx"] / 2.0, b["sy"] / 2.0)))
            for b in buildings]

def _rect_clearance(r_I, buildings):
    """True axis-aligned clearance from the drone to the nearest building edge."""
    min_cl = float("inf")
    for b in buildings:
        hx = b["sx"] / 2.0; hy = b["sy"] / 2.0
        dx = max(abs(r_I[0] - b["cx"]) - hx, 0.0)
        dy = max(abs(r_I[1] - b["cy"]) - hy, 0.0)
        min_cl = min(min_cl, float(np.hypot(dx, dy)))
    return min_cl

def _rect_collision(r_I, buildings, margin=0.0):
    """True if the drone centre is inside any building (with optional margin)."""
    for b in buildings:
        hx = b["sx"] / 2.0 + margin; hy = b["sy"] / 2.0 + margin
        if abs(r_I[0] - b["cx"]) <= hx and abs(r_I[1] - b["cy"]) <= hy:
            return True
    return False

def build_buildings_urban_corridor(start, goal, rng):
    """
    """
    fwd = goal[:2] - start[:2]
    fwd_len = float(np.linalg.norm(fwd))
    fwd_hat = fwd / (fwd_len + 1e-9)
    perp    = np.array([-fwd_hat[1], fwd_hat[0]])

    street_half = 1.8        # half-width of navigable gap [m]
    n_per_side  = 3
    buildings   = []

    for side, sign in enumerate([-1, 1]):
        for k in range(n_per_side):
            t      = (k + 0.5) / n_per_side          # fractional position along corridor
            centre = start[:2] + fwd_hat * (t * fwd_len)
            sx     = float(rng.uniform(2.5, 4.5))    # building depth (along corridor)
            sy     = float(rng.uniform(1.5, 3.0))    # building width (across corridor)
            offset = street_half + sy / 2.0 + float(rng.uniform(0.0, 0.5))
            cx     = float(centre[0] + sign * perp[0] * offset)
            cy     = float(centre[1] + sign * perp[1] * offset)
            side_name = "L" if sign == -1 else "R"
            label  = f"Bldg-{side_name}{k+1}"
            buildings.append({"cx": cx, "cy": cy, "sx": sx, "sy": sy, "label": label})

    obstacles = _rect_to_circle_list(buildings)
    return buildings, obstacles

def build_buildings_urban_dense(start, goal, rng):

    fwd     = goal[:2] - start[:2]
    fwd_len = float(np.linalg.norm(fwd))
    fwd_hat = fwd / (fwd_len + 1e-9)
    perp    = np.array([-fwd_hat[1], fwd_hat[0]])

    n_cols   = 5
    n_rows   = 2
    buildings = []

    for col in range(n_cols):
        t      = (col + 0.5) / n_cols
        c_base = start[:2] + fwd_hat * (t * fwd_len)
        for row, sign in enumerate([-1, 1]):
            sx    = float(rng.uniform(1.8, 3.2))
            sy    = float(rng.uniform(1.5, 2.5))
            # Stagger rows slightly so no perfect straight-line gap exists
            stagger = float(rng.uniform(-0.6, 0.6))
            offset  = 2.2 + sy / 2.0
            cx = float(c_base[0] + sign * perp[0] * offset
                       + fwd_hat[0] * stagger)
            cy = float(c_base[1] + sign * perp[1] * offset
                       + fwd_hat[1] * stagger)
            row_name = chr(ord("A") + row)
            label    = f"Block-{row_name}{col+1}"
            buildings.append({"cx": cx, "cy": cy, "sx": sx, "sy": sy, "label": label})

    obstacles = _rect_to_circle_list(buildings)
    return buildings, obstacles


def sample_start_goal(scenario, rng):
    lo, hi = ARENA
    # Fixed corridor for TC-10, TC-11, TC-12, TC-13 so obstacle layouts are meaningful
    if scenario in ("dynamic_obstacles", "actuator_fault_sweep",
                    "urban_corridor", "urban_dense"):
        return np.array([-7.0, 0.0, 0.0]), np.array([7.0, 0.0, 0.0])
    if scenario == "close_range":
        start = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
        for _ in range(200):
            a = float(rng.uniform(0, 2*np.pi)); d = float(rng.uniform(3.0,5.0))
            goal = start + np.array([np.cos(a)*d, np.sin(a)*d, 0.0])
            if lo < goal[0] < hi and lo < goal[1] < hi:
                return start, goal
        return start, start + np.array([4.0,0.0,0.0])
    if scenario == "long_range":
        for _ in range(500):
            start = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
            goal  = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
            if 12.0 <= np.linalg.norm(goal[:2]-start[:2]) <= 18.0:
                return start, goal
        return np.array([-9.,-9.,0.]), np.array([9.,9.,0.])
    start = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
    for _ in range(200):
        goal = np.array([float(rng.uniform(lo,hi)), float(rng.uniform(lo,hi)), 0.0])
        if np.linalg.norm(goal[:2]-start[:2]) >= 3.0:
            return start, goal
    return start, start + np.array([6.0,0.0,0.0])


# ── Metric helpers ─────────────────────────────────────────────────────────────
def compute_settling_time(dists, goal_radius, dt=DT):
    for i in range(len(dists)):
        if dists[i] < goal_radius and all(d < goal_radius for d in dists[i:]):
            return i * dt
    return -1.0

def compute_overshoot(dists, goal_dist_init):
    if not dists:
        return 0.0
    mi = int(np.argmin(dists))
    if mi >= len(dists)-1:
        return 0.0
    return (max(dists[mi:]) - dists[mi]) / max(goal_dist_init, 1e-6) * 100.0

def compute_steady_state_error(dists, dt=DT, window_s=1.0):
    nw = int(window_s/dt)
    return float(np.mean(np.abs(dists[-nw:]))) if len(dists) >= nw else float(np.mean(np.abs(dists)))

def compute_action_smoothness(actions):
    if len(actions) < 2:
        return 0.0
    arr = np.array(actions)
    return float(np.mean(np.sum(np.diff(arr, axis=0)**2, axis=1)))

def compute_subgoal_consistency(subgoals):
    if len(subgoals) < 2:
        return 1.0
    cs = []
    for i in range(1, len(subgoals)):
        a, b = subgoals[i-1], subgoals[i]
        cs.append(float(np.dot(a,b) / ((np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8))))
    return float(np.mean(cs))

def compute_lyapunov(positions, velocities, goal, alpha=LYAPUNOV_ALPHA, eps=0.01):
    V = [0.5*np.linalg.norm(r[:2]-goal[:2])**2 + alpha*0.5*np.linalg.norm(v[:2])**2
         for r,v in zip(positions, velocities)]
    V  = np.array(V)
    if len(V) < 2:
        return V, 0, 0.0
    dV = np.diff(V)
    return V, int(np.sum(dV > eps)), float(np.max(dV)) if len(dV) else 0.0


# ── TC-10 helpers ─────────────────────────────────────────────────────────────
def _any_dyn_obs_in_range(r_I, dyn_obs):
    return any(
        np.linalg.norm(r_I[:2] - np.array([c[0],c[1]])) - r < DYN_OBS_DETECT_DIST
        for (c, r) in dyn_obs
    )

def _subgoal_direction_changed(subgoals, step, window=5, threshold=0.3):
    """True if subgoal at `step` has shifted >threshold in cosine distance from
    the mean of the previous `window` steps. Detects first avoidance action."""
    if step < window or step >= len(subgoals):
        return False
    prev = np.array(subgoals[max(0,step-window):step]).mean(axis=0)
    curr = subgoals[step]
    cos  = float(np.dot(prev,curr) / ((np.linalg.norm(prev)+1e-8)*(np.linalg.norm(curr)+1e-8)))
    return cos < (1.0 - threshold)


# ── Single trial ───────────────────────────────────────────────────────────────
def run_trial(
    scenario: str,
    trial_idx: int,
    seed: int,
    verbose: bool = False,
    save_timeseries: bool = False,
    ts_dir: str = "benchmark_results/timeseries",
    wind_force: Optional[np.ndarray] = None,
    wind_gust_std: float = 0.0,
    lidar_noise_std: float = 0.0,
    gps_noise_std: float = 0.0,
    vel_noise_std: float = 0.0,
    mass_scale: float = 1.0,
    inertia_scale: float = 1.0,
    drag_scale: float = 1.0,
    kt_scale: float = 1.0,
    fault_motor: int = -1,
    fault_effectiveness: float = 1.0,
    fault_time: float = 0.0,
    nominal_rmse: float = -1.0,
) -> TrialResult:
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    # TC-10 enforces 15 s cap
    max_steps = DYN_OBS_MAX_STEPS if scenario == "dynamic_obstacles" else MAX_STEPS

    params = dynamics.QuadrotorParams()
    params.mass   *= mass_scale
    params.J      *= inertia_scale
    params.cd_lin *= drag_scale
    params.cd_rot *= drag_scale
    params.kT     *= kt_scale
    dyn = dynamics.QuadrotorDynamics(params)

    ctrl = UHRCController(MODEL_PATH, STATS_PATH, device="cpu")
    ctrl.reset()
    _patch_controller_for_preclamp(ctrl)

    start, goal = sample_start_goal(scenario, rng)

    # Build obstacle sets
    # TC-10: 4 static obstacles (always visible) + 2 fixed dynamic ones (hidden until t=3s)
    # TC-12/13: rectangular building layouts; stored separately for true-box clearance logging
    dyn_obstacles: list = []
    rect_buildings: list = []   # populated for urban scenarios only
    if scenario == "dynamic_obstacles":
        obstacles = build_obstacles_random(OBS_COUNT[scenario], start, goal, rng)
        dyn_obstacles = [((cx, cy, 0.0), DYN_OBS_RADIUS) for (cx,cy,_) in DYN_OBS_POSITIONS]
    elif scenario == "narrow_corridor":
        obstacles = build_obstacles_narrow_corridor(start, goal, rng)
    elif scenario == "gap_navigation":
        obstacles = build_obstacles_gap(start, goal, rng)
    elif scenario == "urban_corridor":
        rect_buildings, obstacles = build_buildings_urban_corridor(start, goal, rng)
    elif scenario == "urban_dense":
        rect_buildings, obstacles = build_buildings_urban_dense(start, goal, rng)
    else:
        obstacles = build_obstacles_random(OBS_COUNT.get(scenario,0), start, goal, rng)

    goal_dist_init = float(np.linalg.norm(goal[:2]-start[:2]))
    x_curr = dyn.pack_state(start, np.zeros(3), np.array([1.,0.,0.,0.]), np.zeros(3), np.zeros(4))

    # Accumulators
    positions         = [start.copy()]
    velocities        = [np.zeros(3)]
    dists             = []
    actions_post      = []
    actions_pre       = []
    subgoals          = []
    carry_zh_norms    = []
    carry_zl_norms    = []
    obs_clearances    = []
    dyn_obs_clearances= []
    fz_log            = []
    euler_log         = []

    t_sim      = 0.0
    success    = False
    collision  = False
    conv_step  = -1
    fault_mask = None

    # TC-10 state
    dyn_obs_active        = False
    dyn_detected_step     = -1
    replanning_step       = -1
    dyn_near_miss_count   = 0

    for step in range(max_steps):
        r_I, v_I, q_BI, w_B, Omega = dyn.unpack_state(x_curr)
        psi = float(quat_euler.euler_from_q(q_BI)[2])

        euler = quat_euler.euler_from_q(q_BI)   # [roll, pitch, yaw] rad
        euler_log.append(euler)

        # TC-10: unlock dynamic obstacles in sensor at t>=3s
        if scenario == "dynamic_obstacles" and t_sim >= DYN_OBS_APPEAR_TIME:
            dyn_obs_active = True

        visible_obs  = list(obstacles) + (dyn_obstacles if dyn_obs_active   else [])
        physical_obs = list(obstacles) + (dyn_obstacles if t_sim >= DYN_OBS_APPEAR_TIME else [])

        # Sensor
        lidar = get_lidar_scan(r_I, psi, visible_obs, num_rays=32).copy()
        if lidar_noise_std > 0:
            lidar = np.clip(lidar + rng.normal(0, lidar_noise_std, lidar.shape), 0.0, 5.0)
        r_I_obs = r_I + rng.normal(0, gps_noise_std, 3) if gps_noise_std > 0 else r_I.copy()
        v_I_obs = v_I + rng.normal(0, vel_noise_std, 3) if vel_noise_std > 0 else v_I.copy()

        u_nn, sub_nn = ctrl.get_action(r_I_obs, v_I_obs, q_BI, w_B, lidar, goal)

        actions_post.append(u_nn.copy())
        actions_pre.append(ctrl._last_raw_action.copy())
        subgoals.append(sub_nn.copy())
        carry_zh_norms.append(ctrl._last_carry_norms[0])
        carry_zl_norms.append(ctrl._last_carry_norms[1])
        fz_log.append(float(u_nn[0]))

        dist = float(np.linalg.norm(r_I[:2]-goal[:2]))
        dists.append(dist)
        if conv_step == -1 and dist < CONVERGE_THRESH:
            conv_step = step

        if obstacles:
            # Urban scenarios: use true axis-aligned box clearance for accuracy.
            # All other scenarios: use circle proxy as before.
            if rect_buildings:
                obs_clearances.append(_rect_clearance(r_I, rect_buildings))
            else:
                obs_clearances.append(min(
                    float(np.linalg.norm(r_I[:2]-np.array([c[0],c[1]]))-r_obs)
                    for (c,r_obs) in obstacles
                ))

        # TC-10: dynamic clearance, detection, replanning latency
        if dyn_obs_active and dyn_obstacles:
            dcls = [float(np.linalg.norm(r_I[:2]-np.array([c[0],c[1]]))-r_obs)
                    for (c,r_obs) in dyn_obstacles]
            dyn_obs_clearances.append(min(dcls))
            if min(dcls) < NEAR_MISS_THRESH:
                dyn_near_miss_count += 1
            if dyn_detected_step == -1 and _any_dyn_obs_in_range(r_I, dyn_obstacles):
                dyn_detected_step = step
            if (dyn_detected_step != -1 and replanning_step == -1
                    and _subgoal_direction_changed(subgoals, step)):
                replanning_step = step

        if verbose and step % 100 == 0:
            dstr = f"  dyn={dyn_obs_active}" if scenario == "dynamic_obstacles" else ""
            print(f"  [{scenario}] t={trial_idx} step={step:4d}"
                  f"  pos=({r_I[0]:6.2f},{r_I[1]:6.2f})  dist={dist:.2f}m"
                  f"  Fz={u_nn[0]:.3f}  sub=({sub_nn[0]:+.2f},{sub_nn[1]:+.2f})"
                  f"  ||zH||={ctrl._last_carry_norms[0]:.1f}{dstr}")

        # Wind
        u_applied = u_nn.copy()
        if wind_force is not None:
            R_BI = quat_euler.R_BI_from_q(q_BI)
            wb   = R_BI @ wind_force
            gb   = R_BI @ rng.normal(0, wind_gust_std, 3) if wind_gust_std > 0 else np.zeros(3)
            u_applied[0] += wb[2] + gb[2]

        # Actuator fault
        if fault_motor >= 0 and t_sim >= fault_time:
            if fault_mask is None:
                fault_mask = np.ones(4)
            fault_mask[fault_motor] = fault_effectiveness

        x_curr = _rk4_step(dyn, t_sim, x_curr, u_applied, fault_mask=fault_mask)
        t_sim += DT

        r_new, v_new, *_ = dyn.unpack_state(x_curr)
        positions.append(r_new.copy())
        velocities.append(v_new.copy())

        # Collision against all physical obstacles (dynamic included after t=3s).
        # Urban scenarios use true axis-aligned box test; others use circle proxy.
        if rect_buildings:
            hit = _rect_collision(r_new, rect_buildings)
        else:
            hit = any(float(np.linalg.norm(r_new[:2]-np.array([c[0],c[1]])))<float(r_obs)
                      for c,r_obs in physical_obs)
        reached = float(np.linalg.norm(r_new[:2]-goal[:2])) < GOAL_RADIUS
        if hit:
            collision = True; break
        if reached:
            success   = True; break

    # ── Post-hoc metrics ───────────────────────────────────────────────────
    steps_taken = len(dists)
    path        = np.array(positions)
    dists_arr   = np.array(dists, dtype=float)
    time_arr    = np.arange(len(dists)) * DT

    rmse = float(np.sqrt(np.mean(dists_arr**2))) if len(dists_arr) else 0.0
    iae  = float(np.sum(np.abs(dists_arr)) * DT)  if len(dists_arr) else 0.0
    ise  = float(np.sum(dists_arr**2)      * DT)  if len(dists_arr) else 0.0
    itae = float(np.sum(time_arr*np.abs(dists_arr)) * DT) if len(dists_arr) else 0.0

    # ITAE post-event (TC-10 primary metric)
    event_step = int(DYN_OBS_APPEAR_TIME / DT)
    if len(dists_arr) > event_step:
        pe_d = dists_arr[event_step:]; pe_t = time_arr[event_step:]
        itae_post = float(np.sum(pe_t * np.abs(pe_d)) * DT)
    else:
        itae_post = float("nan")

    pos_arr = path[:-1,:2] if len(path) > 1 else path[:,:2]
    rmse_x  = float(np.sqrt(np.mean((pos_arr[:,0]-goal[0])**2))) if len(pos_arr) else 0.0
    rmse_y  = float(np.sqrt(np.mean((pos_arr[:,1]-goal[1])**2))) if len(pos_arr) else 0.0

    path_length = float(np.sum(np.linalg.norm(np.diff(path[:,:2], axis=0), axis=1)))
    plr         = path_length / max(goal_dist_init, 1e-6)
    near_miss   = int(np.sum(np.array(obs_clearances) < NEAR_MISS_THRESH)) if obs_clearances else 0

    settling   = compute_settling_time(dists, CONVERGE_THRESH)
    overshoot  = compute_overshoot(dists, goal_dist_init)
    ss_error   = compute_steady_state_error(dists) if success else -1.0
    smoothness = compute_action_smoothness(actions_post)
    sg_consist = compute_subgoal_consistency(subgoals)

    fz_arr   = np.array(fz_log) if fz_log else np.array([HOVER_THRUST])
    fz_mean  = float(np.mean(fz_arr))
    fz_max   = float(np.max(fz_arr))
    fz_drift = float(np.mean(np.abs(fz_arr-HOVER_THRUST)))

    final_dist = float(np.linalg.norm(path[-1,:2]-goal[:2]))
    min_dist   = float(np.min(dists_arr)) if len(dists_arr) else goal_dist_init
    mean_dist  = float(np.mean(dists_arr)) if len(dists_arr) else goal_dist_init
    min_obs_cl = float(min(obs_clearances)) if obs_clearances else -1.0

    V_ser, lyap_viol, lyap_mdv = compute_lyapunov(
        positions[:len(dists)], velocities[:len(dists)], goal)
    lyap_vf = float(V_ser[-1]) if len(V_ser) else 0.0

    zh_arr = np.array(carry_zh_norms) if carry_zh_norms else np.array([0.0])
    zl_arr = np.array(carry_zl_norms) if carry_zl_norms else np.array([0.0])

    pre_arr = np.array(actions_pre); post_arr = np.array(actions_post)
    if len(pre_arr) and len(post_arr):
        cmask      = np.any(np.abs(pre_arr-post_arr) > 1e-6, axis=1)
        clamp_acts = int(np.sum(cmask))
        max_consec = cur = 0
        for c in cmask:
            cur = cur+1 if c else 0
            max_consec = max(max_consec, cur)
    else:
        clamp_acts = max_consec = 0

    # TC-10 final values
    if scenario == "dynamic_obstacles":
        dyn_min_cl = float(min(dyn_obs_clearances)) if dyn_obs_clearances else -1.0
        rep_lat    = (float((replanning_step - dyn_detected_step)*DT)
                      if replanning_step > 0 and dyn_detected_step >= 0 else -1.0)
    else:
        dyn_min_cl = -1.0; rep_lat = -1.0; dyn_near_miss_count = 0; itae_post = float("nan")

    # TC-11 attitude metrics
    if euler_log:
        ea        = np.array(euler_log)
        max_roll  = float(np.max(np.abs(ea[:,0])) * 180.0/np.pi)
        max_pitch = float(np.max(np.abs(ea[:,1])) * 180.0/np.pi)
    else:
        max_roll = max_pitch = 0.0

    rmse_delta  = (rmse - nominal_rmse) if nominal_rmse >= 0.0 else -1.0
    fault_sev   = fault_effectiveness if fault_motor >= 0 else 1.0

    # Save time-series
    if save_timeseries:
        Path(ts_dir).mkdir(parents=True, exist_ok=True)
        ts_save = dict(
            positions=np.array(positions), velocities=np.array(velocities),
            dists=dists_arr, actions_post=post_arr, actions_pre=pre_arr,
            subgoals=np.array(subgoals), carry_zh_norms=zh_arr, carry_zl_norms=zl_arr,
            obs_clearances=np.array(obs_clearances),
            dyn_obs_clearances=np.array(dyn_obs_clearances),
            euler_log=np.array(euler_log) if euler_log else np.zeros((0,3)),
            lyapunov_V=V_ser, goal=goal, start=start,
            obstacles=np.array([(c[0],c[1],r) for c,r in obstacles]) if obstacles else np.array([]),
            dyn_obstacles=np.array([(c[0],c[1],r) for c,r in dyn_obstacles]) if dyn_obstacles else np.array([]),
        )
        if rect_buildings:
            # Save building geometry and labels for post-hoc visualisation
            ts_save["building_cx"]     = np.array([b["cx"]    for b in rect_buildings])
            ts_save["building_cy"]     = np.array([b["cy"]    for b in rect_buildings])
            ts_save["building_sx"]     = np.array([b["sx"]    for b in rect_buildings])
            ts_save["building_sy"]     = np.array([b["sy"]    for b in rect_buildings])
            ts_save["building_labels"] = np.array([b["label"] for b in rect_buildings])
        np.savez_compressed(
            os.path.join(ts_dir, f"{scenario}_t{trial_idx:03d}_s{seed}.npz"),
            **ts_save,
        )

    return TrialResult(
        scenario=scenario, trial_idx=trial_idx, seed=seed,
        n_obstacles=len(obstacles)+len(dyn_obstacles), goal_dist_init=goal_dist_init,
        success=success, collision=collision, timeout=not success and not collision,
        final_dist=final_dist, min_dist=min_dist, mean_dist=mean_dist,
        rmse=rmse, iae=iae, ise=ise, itae=itae,
        path_length=path_length, path_length_ratio=plr,
        near_miss_count=near_miss, completion_time_s=steps_taken*DT,
        convergence_step=conv_step, settling_time_s=settling,
        overshoot=overshoot, steady_state_error=ss_error,
        action_smoothness=smoothness, subgoal_consistency=sg_consist,
        fz_mean=fz_mean, fz_max=fz_max, fz_drift=fz_drift, steps_taken=steps_taken,
        min_obs_clearance=min_obs_cl,
        lyapunov_v_final=lyap_vf, lyapunov_violations=lyap_viol, lyapunov_max_dv=lyap_mdv,
        carry_norm_zh_max=float(np.max(zh_arr)), carry_norm_zl_max=float(np.max(zl_arr)),
        carry_norm_zh_final=float(zh_arr[-1]),   carry_norm_zl_final=float(zl_arr[-1]),
        clamp_activations=clamp_acts, max_consecutive_clamps=max_consec,
        rmse_x=rmse_x, rmse_y=rmse_y,
        replanning_latency_s=rep_lat, dyn_obs_near_miss_count=dyn_near_miss_count,
        dyn_obs_min_clearance=dyn_min_cl, itae_post_event=itae_post,
        fault_severity=fault_sev, max_roll_deg=max_roll, max_pitch_deg=max_pitch,
        rmse_nominal_delta=rmse_delta, wall_time_s=time.perf_counter()-t0,
    )


# ── TC-11 orchestrator ─────────────────────────────────────────────────────────
def run_fault_sweep(n_trials, seed_base, verbose, save_timeseries, ts_dir, out_dir):
    """
    Runs nominal baseline + Motor 1 at 70/50/30% effectiveness.
    Saves fault_sweep_summary.csv and fault_sweep_plot.png to out_dir.
    Returns flat list of all TrialResult objects.
    """
    print(f"\n  ▶ TC-11  actuator_fault_sweep"
          f"  (Motor {FAULT_SWEEP_MOTOR}, t_fault={FAULT_SWEEP_TIME}s,"
          f" levels={[int(l*100) for l in FAULT_SWEEP_LEVELS]}%, n={n_trials})")

    all_trials: list[TrialResult] = []
    sweep_rows: list[dict] = []
    done = 0

    # Nominal baseline
    nom: list[TrialResult] = []
    print(f"    [Nominal — no fault]")
    for i in range(n_trials):
        tr = run_trial("actuator_fault_sweep", i, seed_base+done,
                       verbose=verbose, save_timeseries=save_timeseries, ts_dir=ts_dir,
                       fault_motor=-1, nominal_rmse=-1.0)
        nom.append(tr); all_trials.append(tr); done += 1
    nom_rmse = float(np.mean([t.rmse for t in nom]))
    nom_succ = float(np.mean([t.success for t in nom]))
    print(f"    Nominal: succ={nom_succ*100:.1f}%  RMSE={nom_rmse:.4f}")
    sweep_rows.append({"fault_severity": 1.0, "success_rate": nom_succ,
                       "mean_rmse": nom_rmse, "rmse_delta": 0.0,
                       "mean_max_roll_deg":  float(np.mean([t.max_roll_deg  for t in nom])),
                       "mean_max_pitch_deg": float(np.mean([t.max_pitch_deg for t in nom]))})

    # Fault levels
    for eff in FAULT_SWEEP_LEVELS:
        lvl: list[TrialResult] = []
        label = f"{int(eff*100)}%"
        print(f"    [Motor {FAULT_SWEEP_MOTOR} @ {label}]")
        for i in range(n_trials):
            tr = run_trial("actuator_fault_sweep", i, seed_base+done,
                           verbose=verbose, save_timeseries=save_timeseries, ts_dir=ts_dir,
                           fault_motor=FAULT_SWEEP_MOTOR, fault_effectiveness=eff,
                           fault_time=FAULT_SWEEP_TIME, nominal_rmse=nom_rmse)
            lvl.append(tr); all_trials.append(tr); done += 1
            icon = "✅" if tr.success else ("💥" if tr.collision else "⏱")
            print(f"      {i+1:3d}/{n_trials} {icon}"
                  f"  RMSE={tr.rmse:.3f}  delta={tr.rmse_nominal_delta:+.3f}"
                  f"  roll={tr.max_roll_deg:.1f}°  pitch={tr.max_pitch_deg:.1f}°",
                  flush=True)
        s_rmse  = float(np.mean([t.rmse for t in lvl]))
        s_succ  = float(np.mean([t.success for t in lvl]))
        s_roll  = float(np.mean([t.max_roll_deg  for t in lvl]))
        s_pitch = float(np.mean([t.max_pitch_deg for t in lvl]))
        print(f"    @{label}: succ={s_succ*100:.1f}%  RMSE={s_rmse:.4f}"
              f"  ΔRMSE={s_rmse-nom_rmse:+.4f}"
              f"  roll={s_roll:.2f}°  pitch={s_pitch:.2f}°")
        sweep_rows.append({"fault_severity": eff, "success_rate": s_succ,
                           "mean_rmse": s_rmse, "rmse_delta": s_rmse-nom_rmse,
                           "mean_max_roll_deg": s_roll, "mean_max_pitch_deg": s_pitch})

    # Save CSV
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "fault_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader(); w.writerows(sweep_rows)
    print(f"    Saved → {csv_path}")

    # Plot
    _plot_fault_sweep(sweep_rows, out_dir)
    return all_trials


def _plot_fault_sweep(rows, out_dir):
    labels  = [f"{int(r['fault_severity']*100)}%" for r in rows]
    x       = np.arange(len(labels))
    sevs    = [r['fault_severity'] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"TC-11 Actuator Fault Sweep — Motor {FAULT_SWEEP_MOTOR}", fontsize=12)

    axes[0].bar(x, [r['success_rate']*100 for r in rows],
                color=["#4CAF50" if s==1.0 else "#F44336" for s in sevs])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Success (%)"); axes[0].set_title("Success rate"); axes[0].set_ylim(0,110)

    axes[1].bar(x, [r['mean_rmse'] for r in rows], color="#2196F3", alpha=0.85)
    axes[1].axhline(rows[0]['mean_rmse'], color="k", linestyle="--", lw=0.8, label="Nominal")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("RMSE (m)"); axes[1].set_title("RMSE"); axes[1].legend(fontsize=8)

    axes[2].bar(x, [r['rmse_delta'] for r in rows],
                color=["#9E9E9E" if d==0 else "#FF5722" for d in [r['rmse_delta'] for r in rows]])
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("ΔRMSE (m)"); axes[2].set_title("RMSE increase vs nominal")

    w = 0.35
    axes[3].bar(x-w/2, [r['mean_max_roll_deg']  for r in rows], w, label="Roll",  color="#E91E63", alpha=0.85)
    axes[3].bar(x+w/2, [r['mean_max_pitch_deg'] for r in rows], w, label="Pitch", color="#9C27B0", alpha=0.85)
    axes[3].set_xticks(x); axes[3].set_xticklabels(labels)
    axes[3].set_ylabel("Degrees"); axes[3].set_title("Max attitude deviation"); axes[3].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fault_sweep_plot.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"    Saved → {fig_path}")
    plt.close(fig)


# ── Aggregate stats ────────────────────────────────────────────────────────────
def compute_stats(trials: list[TrialResult]) -> ScenarioStats:
    assert trials
    def _m(v): return float(np.mean(v))
    def _s(v): return float(np.std(v))

    clears    = [t.min_obs_clearance for t in trials if t.min_obs_clearance >= 0]
    settle_ok = [t.settling_time_s   for t in trials if t.settling_time_s   >= 0]
    ss_ok     = [t.steady_state_error for t in trials if t.steady_state_error >= 0]
    rep_lats  = [t.replanning_latency_s    for t in trials if t.replanning_latency_s >= 0]
    dyn_clrs  = [t.dyn_obs_min_clearance   for t in trials if t.dyn_obs_min_clearance >= 0]
    itae_post = [t.itae_post_event for t in trials
                 if not (isinstance(t.itae_post_event, float) and np.isnan(t.itae_post_event))]
    rmse_dels = [t.rmse_nominal_delta for t in trials if t.rmse_nominal_delta >= 0]

    return ScenarioStats(
        scenario=trials[0].scenario, n_trials=len(trials),
        n_obstacles=trials[0].n_obstacles,
        success_rate=_m([t.success   for t in trials]),
        collision_rate=_m([t.collision for t in trials]),
        timeout_rate=_m([t.timeout   for t in trials]),
        mean_rmse=_m([t.rmse for t in trials]), std_rmse=_s([t.rmse for t in trials]),
        mean_iae=_m([t.iae  for t in trials]),
        mean_ise=_m([t.ise  for t in trials]),
        mean_itae=_m([t.itae for t in trials]),
        mean_path_length=_m([t.path_length for t in trials]),
        std_path_length=_s([t.path_length  for t in trials]),
        mean_plr=_m([t.path_length_ratio for t in trials]),
        mean_near_miss=_m([t.near_miss_count  for t in trials]),
        mean_completion_time=_m([t.completion_time_s for t in trials]),
        mean_settling_time=float(np.mean(settle_ok)) if settle_ok else float("nan"),
        mean_overshoot=_m([t.overshoot for t in trials]),
        mean_ss_error=float(np.mean(ss_ok)) if ss_ok else float("nan"),
        mean_action_smooth=_m([t.action_smoothness   for t in trials]),
        mean_subgoal_consist=_m([t.subgoal_consistency for t in trials]),
        mean_final_dist=_m([t.final_dist for t in trials]),
        std_final_dist=_s([t.final_dist  for t in trials]),
        mean_min_dist=_m([t.min_dist for t in trials]),
        mean_lyap_violations=_m([t.lyapunov_violations for t in trials]),
        mean_carry_zh_max=_m([t.carry_norm_zh_max for t in trials]),
        mean_carry_zl_max=_m([t.carry_norm_zl_max for t in trials]),
        mean_clamp_acts=_m([t.clamp_activations for t in trials]),
        mean_fz_drift=_m([t.fz_drift for t in trials]),
        std_fz_drift=_s([t.fz_drift  for t in trials]),
        mean_obs_clearance=float(np.mean(clears)) if clears else float("nan"),
        std_obs_clearance=float(np.std(clears))   if clears else float("nan"),
        mean_wall_time_s=_m([t.wall_time_s for t in trials]),
        mean_replanning_latency_s=float(np.mean(rep_lats)) if rep_lats else float("nan"),
        mean_dyn_obs_near_miss=_m([t.dyn_obs_near_miss_count for t in trials]),
        mean_dyn_obs_min_clearance=float(np.mean(dyn_clrs)) if dyn_clrs else float("nan"),
        mean_itae_post_event=float(np.mean(itae_post)) if itae_post else float("nan"),
        fault_severity=float(np.mean([t.fault_severity for t in trials])),
        mean_max_roll_deg=_m([t.max_roll_deg  for t in trials]),
        mean_max_pitch_deg=_m([t.max_pitch_deg for t in trials]),
        mean_rmse_nominal_delta=float(np.mean(rmse_dels)) if rmse_dels else float("nan"),
    )


_TC_IDS = {
    "open_field": "TC-01", "dense": "TC-02", "narrow_corridor": "TC-03",
    "gap_navigation": "TC-04", "close_range": "TC-05", "long_range": "TC-06",
    "dynamic_obstacles": "TC-10", "actuator_fault_sweep": "TC-11",
    "urban_corridor": "TC-12", "urban_dense": "TC-13",
}

def print_summary_table(stats_list):
    sep = "=" * 132
    print(f"\n{sep}\n  UHRC Benchmark Results\n{sep}")
    hdr = (f"  {'TC':6s}  {'Scenario':20s}  {'Succ%':6s}  {'RMSE':7s}  {'IAE':8s}"
           f"  {'ITAE':9s}  {'PLR':5s}  {'Settle':7s}  {'Smooth':7s}"
           f"  {'SubgCos':7s}  {'LyapViol':8s}  {'ClampN':6s}"
           f"  {'RepLat':7s}  {'DynClr':7s}")
    print(hdr); print("-"*132)
    for s in stats_list:
        tc = _TC_IDS.get(s.scenario, "TC-??")
        st = f"{s.mean_settling_time:.2f}" if not np.isnan(s.mean_settling_time) else "  N/A"
        rl = f"{s.mean_replanning_latency_s:.3f}" if not np.isnan(s.mean_replanning_latency_s) else "  N/A"
        dc = f"{s.mean_dyn_obs_min_clearance:.3f}" if not np.isnan(s.mean_dyn_obs_min_clearance) else "  N/A"
        print(f"  {tc:6s}  {s.scenario:20s}  {s.success_rate*100:5.1f}%"
              f"  {s.mean_rmse:6.3f}  {s.mean_iae:7.3f}  {s.mean_itae:8.3f}"
              f"  {s.mean_plr:5.2f}  {st:>7s}  {s.mean_action_smooth:6.4f}"
              f"  {s.mean_subgoal_consist:6.3f}  {s.mean_lyap_violations:7.1f}"
              f"  {s.mean_clamp_acts:5.1f}  {rl:>7s}  {dc:>7s}")
    print(sep)

def save_results(trials, stats, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trial_dicts = [asdict(t) for t in trials]
    if trial_dicts:
        with open(os.path.join(out_dir,"trials.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trial_dicts[0].keys()))
            w.writeheader(); w.writerows(trial_dicts)
        print(f"  Saved per-trial data    → {out_dir}/trials.csv")
    stats_dicts = [asdict(s) for s in stats]
    if stats_dicts:
        with open(os.path.join(out_dir,"summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_dicts[0].keys()))
            w.writeheader(); w.writerows(stats_dicts)
        print(f"  Saved scenario summary  → {out_dir}/summary.csv")
    with open(os.path.join(out_dir,"results.json"), "w") as f:
        json.dump({"trials": trial_dicts, "summary": stats_dicts}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and np.isnan(o) else o)
    print(f"  Saved full JSON         → {out_dir}/results.json")

def plot_summary(stats_list, out_dir):
    # Exclude TC-11 — it has its own dedicated plot
    plot_stats = [s for s in stats_list if s.scenario != "actuator_fault_sweep"]
    if not plot_stats:
        return
    scenarios = [s.scenario.replace("_","\n") for s in plot_stats]
    x = np.arange(len(scenarios)); w = 0.55
    fig, axes = plt.subplots(2, 4, figsize=(22,10))
    fig.suptitle("UHRC Benchmark Summary", fontsize=13)

    ax = axes[0,0]
    succ=[s.success_rate*100   for s in plot_stats]
    coll=[s.collision_rate*100 for s in plot_stats]
    tout=[s.timeout_rate*100   for s in plot_stats]
    ax.bar(x, succ, w, label="Success",   color="#4CAF50")
    ax.bar(x, coll, w, bottom=succ,       label="Collision",color="#F44336")
    ax.bar(x, tout, w, bottom=[a+b for a,b in zip(succ,coll)], label="Timeout",color="#FFC107")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("%"); ax.set_title("Outcomes"); ax.legend(fontsize=7)

    ax = axes[0,1]
    vals=[s.mean_rmse for s in plot_stats]; errs=[s.std_rmse for s in plot_stats]
    ax.bar(x, vals, w, color="#2196F3", alpha=0.85)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE")

    ax = axes[0,2]
    ax.bar(x, [s.mean_itae for s in plot_stats], w, color="#9C27B0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("ITAE"); ax.set_title("ITAE")

    ax = axes[0,3]
    ax.bar(x, [s.mean_action_smooth for s in plot_stats], w, color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("||Δu||²"); ax.set_title("Action smoothness")

    ax = axes[1,0]
    ax.bar(x, [s.mean_lyap_violations for s in plot_stats], w, color="#F44336", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("Violations"); ax.set_title("Lyapunov ΔV violations")

    ax = axes[1,1]
    ax.bar(x-0.15, [s.mean_carry_zh_max for s in plot_stats], 0.3, label="||z_H||", color="#3F51B5", alpha=0.85)
    ax.bar(x+0.15, [s.mean_carry_zl_max for s in plot_stats], 0.3, label="||z_L||", color="#00BCD4", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("Frobenius norm"); ax.set_title("Carry norm max"); ax.legend(fontsize=7)

    ax = axes[1,2]
    rl_vals  = [s.mean_replanning_latency_s if not np.isnan(s.mean_replanning_latency_s) else 0.0 for s in plot_stats]
    rl_cols  = ["#FF5722" if not np.isnan(s.mean_replanning_latency_s) else "#BDBDBD" for s in plot_stats]
    ax.bar(x, rl_vals, w, color=rl_cols, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("Latency (s)"); ax.set_title("TC-10: Replanning latency")

    ax = axes[1,3]
    dc_vals = [s.mean_dyn_obs_min_clearance if not np.isnan(s.mean_dyn_obs_min_clearance) else 0.0 for s in plot_stats]
    dc_cols = ["#FF5722" if not np.isnan(s.mean_dyn_obs_min_clearance) else "#BDBDBD" for s in plot_stats]
    ax.bar(x, dc_vals, w, color=dc_cols, alpha=0.85)
    ax.axhline(NEAR_MISS_THRESH, color="red", linestyle="--", lw=0.8, label=f"Near-miss ({NEAR_MISS_THRESH}m)")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("Min clearance (m)"); ax.set_title("TC-10: Dyn obs clearance"); ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "benchmark_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved summary figure    → {fig_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",        type=str,   default=None)
    p.add_argument("--n",               type=int,   default=DEFAULT_TRIALS)
    p.add_argument("--seed_base",       type=int,   default=0)
    p.add_argument("--out",             type=str,   default="benchmark_results")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--quick",           action="store_true")
    p.add_argument("--save_timeseries", action="store_true")
    p.add_argument("--wind",      nargs=3, type=float, default=None, metavar=("FX","FY","FZ"))
    p.add_argument("--wind_gust",     type=float, default=0.0)
    p.add_argument("--lidar_noise",   type=float, default=0.0)
    p.add_argument("--gps_noise",     type=float, default=0.0)
    p.add_argument("--vel_noise",     type=float, default=0.0)
    p.add_argument("--mass_scale",    type=float, default=1.0)
    p.add_argument("--inertia_scale", type=float, default=1.0)
    p.add_argument("--drag_scale",    type=float, default=1.0)
    p.add_argument("--kt_scale",      type=float, default=1.0)
    p.add_argument("--fault_motor",   type=int,   default=-1)
    p.add_argument("--fault_eff",     type=float, default=1.0)
    p.add_argument("--fault_time",    type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    all_scenarios = list(OBS_COUNT.keys())

    VALID_SCENARIOS = list(OBS_COUNT.keys()) + ["actuator_fault_sweep"]

    if args.quick:
        scenarios = all_scenarios[:3]; n_trials = 10
    elif args.scenario:
        if args.scenario not in VALID_SCENARIOS:
            print(f"Unknown scenario. Choose from: {VALID_SCENARIOS}"); return
        scenarios = [args.scenario]; n_trials = args.n
    else:
        scenarios = all_scenarios + ["actuator_fault_sweep"]; n_trials = args.n

    wind = np.array(args.wind) if args.wind else None

    print(f"\n{'='*72}")
    print(f"  UHRC Benchmark  |  {len(scenarios)} scenario(s)  |  {n_trials} trials each")
    print(f"  Model : {MODEL_PATH}")
    if wind is not None:
        print(f"  Wind  : F={wind}N  gust_std={args.wind_gust}N")
    if args.lidar_noise > 0 or args.gps_noise > 0:
        print(f"  Noise : LiDAR={args.lidar_noise}m  GPS={args.gps_noise}m  vel={args.vel_noise}m/s")
    if args.mass_scale != 1.0 or args.inertia_scale != 1.0:
        print(f"  Params: mass*{args.mass_scale}  J*{args.inertia_scale}"
              f"  drag*{args.drag_scale}  kT*{args.kt_scale}")
    if args.fault_motor >= 0:
        print(f"  Fault : motor {args.fault_motor} at {args.fault_eff*100:.0f}% from t={args.fault_time}s")
    print(f"{'='*72}\n")

    all_trials: list[TrialResult] = []
    all_stats:  list[ScenarioStats] = []
    done = 0

    for sc in scenarios:
        # TC-11 has its own multi-level orchestrator
        if sc == "actuator_fault_sweep":
            fault_trials = run_fault_sweep(
                n_trials=n_trials, seed_base=args.seed_base+done,
                verbose=args.verbose, save_timeseries=args.save_timeseries,
                ts_dir=os.path.join(args.out,"timeseries"), out_dir=args.out,
            )
            all_trials.extend(fault_trials)
            # Build per-severity stats for inclusion in main summary CSV
            sev_map: dict[float, list] = {}
            for tr in fault_trials:
                sev_map.setdefault(tr.fault_severity, []).append(tr)
            for sev, tlist in sorted(sev_map.items(), reverse=True):
                all_stats.append(compute_stats(tlist))
            done += n_trials * (1 + len(FAULT_SWEEP_LEVELS))
            continue

        print(f"  ▶ {_TC_IDS.get(sc,'??')} {sc}  (obs={OBS_COUNT[sc]}, n={n_trials})")
        sc_trials: list[TrialResult] = []
        grand = len(scenarios) * n_trials

        for i in range(n_trials):
            tr = run_trial(
                sc, i, args.seed_base+done,
                verbose=args.verbose, save_timeseries=args.save_timeseries,
                ts_dir=os.path.join(args.out,"timeseries"),
                wind_force=wind, wind_gust_std=args.wind_gust,
                lidar_noise_std=args.lidar_noise, gps_noise_std=args.gps_noise,
                vel_noise_std=args.vel_noise, mass_scale=args.mass_scale,
                inertia_scale=args.inertia_scale, drag_scale=args.drag_scale,
                kt_scale=args.kt_scale, fault_motor=args.fault_motor,
                fault_effectiveness=args.fault_eff, fault_time=args.fault_time,
            )
            sc_trials.append(tr); done += 1
            icon  = "✅" if tr.success else ("💥" if tr.collision else "⏱")
            extra = ""
            if sc == "dynamic_obstacles":
                rl = f"{tr.replanning_latency_s:.3f}s" if tr.replanning_latency_s >= 0 else "N/A"
                extra = f"  repLat={rl}  dynClr={tr.dyn_obs_min_clearance:.2f}m"
            print(f"    [{done:4d}] trial {i+1:3d}/{n_trials}  {icon}"
                  f"  RMSE={tr.rmse:.3f}  IAE={tr.iae:.2f}"
                  f"  clamp={tr.clamp_activations}"
                  f"  ||zH||={tr.carry_norm_zh_max:.1f}{extra}", flush=True)

        stats = compute_stats(sc_trials)
        all_trials.extend(sc_trials); all_stats.append(stats)

        extra_summary = ""
        if sc == "dynamic_obstacles":
            rl = f"{stats.mean_replanning_latency_s:.3f}" if not np.isnan(stats.mean_replanning_latency_s) else "N/A"
            extra_summary = (f"  repLat={rl}s"
                             f"  dynNM={stats.mean_dyn_obs_near_miss:.1f}"
                             f"  dynClr={stats.mean_dyn_obs_min_clearance:.3f}m"
                             f"  ITAE_post={stats.mean_itae_post_event:.3f}")
        print(f"    → succ={stats.success_rate*100:.1f}%"
              f"  RMSE={stats.mean_rmse:.3f}  ITAE={stats.mean_itae:.3f}"
              f"  lyap={stats.mean_lyap_violations:.1f}{extra_summary}\n")

    if all_stats:
        print_summary_table(all_stats)
    save_results(all_trials, all_stats, args.out)
    if all_stats:
        plot_summary(all_stats, args.out)

    if all_trials:
        ns = sum(t.success for t in all_trials)
        print(f"\n  Overall: {ns}/{len(all_trials)} ({ns/len(all_trials)*100:.1f}%) successful.\n")


if __name__ == "__main__":
    main()