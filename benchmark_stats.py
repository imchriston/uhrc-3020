"""
benchmark_test.py
Structured Monte-Carlo benchmark for the UHRC controller.


Active test cases
  TC-01  open_field          — 0 obstacles, pure setpoint tracking
  TC-02  narrow_corridor     — 4 obstacles forming a tight gap
  TC-03  gap_navigation      — 4 obstacles arranged around a single gap
  TC-04  dynamic_obstacles   — 2 static + 3 pop-in dynamic obstacles



Usage
-----
  python benchmark_test.py                          # full active benchmark
  python benchmark_test.py --scenario narrow_corridor --n 30
  python benchmark_test.py --save_timeseries        # save per-trial .npz

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

import dynamics
from generate_data_sensors import get_lidar_scan
from uhrc_ctrl_wp import UHRCController
import utils.quat_euler as quat_euler


MODEL_PATH = "checkpoints/uhrc_best_waypoint.pth"
STATS_PATH = "checkpoints/norm_stats_waypoint.npz"

DT               = 0.01
MAX_STEPS        = 1500
GOAL_RADIUS      = 0.5
CONVERGE_THRESH  = 1.0
HOVER_Z          = 1.0
ARENA            = (-10.0, 10.0)
DEFAULT_TRIALS   = 30
NEAR_MISS_THRESH = 0.3          
LIDAR_MAX        = 5.0

# Dynamic obstacle constants
DYN_OBS_N_STATIC  = 2           # static background obstacles
DYN_OBS_N_DYNAMIC = 3           # pop-in obstacles


ACTIVE_SCENARIOS = {
    "open_field":      0,
    "narrow_corridor": 4,
    "gap_navigation":  4,
    "dynamic_obstacles": DYN_OBS_N_STATIC,
}



_TC_IDS = {
    "open_field":      "TC-01",
    # "dense":           "TC-02",
    "narrow_corridor": "TC-03",
    "gap_navigation":  "TC-04",
    # "close_range":     "TC-05",
    # "long_range":      "TC-06",
    "dynamic_obstacles":      "TC-10",
    # "actuator_fault_sweep":   "TC-11",
    # "urban_corridor":         "TC-12",
    # "urban_dense":            "TC-13",
}


#  DYNAMIC OBSTACLE DATACLASS 

@dataclass
class DynamicObstacle:
    """
    A circular obstacle that pops into the sensor field at a fixed simulation
    step.  Before `appear_step` it is invisible to the lidar and does not
    trigger collisions; afterwards both apply.
    """
    cx: float
    cy: float
    r: float
    appear_step: int

    def is_active(self, step: int) -> bool:
        return step >= self.appear_step

    def as_circle_tuple(self) -> Tuple[Tuple, float]:
        """Return the ((cx, cy, 0.0), r) format used by get_lidar_scan."""
        return ((self.cx, self.cy, 0.0), self.r)


#  OBSTACLE BUILDERS

def _safe_obs(cx: float, cy: float, r: float, pts, gap: float = 1.0) -> bool:
    """True if the candidate circle is far enough from all reference points."""
    return all(
        np.linalg.norm(np.array([cx, cy]) - np.array(pt[:2])) >= r + gap
        for pt in pts
    )

def _min_sep(centers, candidate, radii, r_cand, margin: float = 0.4) -> bool:
    """True if the candidate does not overlap any existing circle + margin."""
    return all(
        np.linalg.norm(np.array(c) - np.array(candidate)) >= r + r_cand + margin
        for c, r in zip(centers, radii)
    )


def build_static_circles(n: int, start, goal, rng) -> List[Tuple]:
    """Place n random circular obstacles clear of start and goal."""
    obs, centers, radii = [], [], []
    for _ in range(500):
        if len(obs) >= n:
            break
        cx = float(rng.uniform(*ARENA))
        cy = float(rng.uniform(*ARENA))
        r  = float(rng.uniform(0.5, 1.0))
        if not _safe_obs(cx, cy, r, [start, goal]):
            continue
        if not _min_sep(centers, (cx, cy), radii, r):
            continue
        obs.append(((cx, cy, 0.0), r))
        centers.append((cx, cy))
        radii.append(r)
    return obs


def build_static_circles_general(n: int, start, goal, rng,
                                  r_lo: float = 0.5, r_hi: float = 1.2) -> List[Tuple]:
    """Generic random obstacle builder (used by open_field if ever n>0)."""
    return build_static_circles(n, start, goal, rng)


def build_dynamic_pop_obstacles(
        n: int, start, goal, rng,
        static_circles: List[Tuple],
        max_step: int) -> List[DynamicObstacle]:
    """
    Place n dynamic obstacles along the start→goal corridor at fractional
    positions and assign uniformly-sampled appearance steps within the first
    half of the expected flight duration.  Mirrors benchmark_dynamic.py.
    """
    fwd  = goal[:2] - start[:2]
    dist = float(np.linalg.norm(fwd)) + 1e-9
    fwd  = fwd / dist
    perp = np.array([-fwd[1], fwd[0]])

    st_centers = [c[:2] for c, _ in static_circles]
    st_radii   = [r     for _, r in static_circles]

    dyn_obs, centers, radii = [], [], []
    fracs = [0.35, 0.52, 0.68]

    for i in range(n):
        frac = fracs[i % len(fracs)]
        for _ in range(300):
            lat = float(rng.uniform(-1.0, 1.0))
            r   = float(rng.uniform(0.45, 0.75))
            cx  = float(start[0] + fwd[0] * dist * frac + perp[0] * lat)
            cy  = float(start[1] + fwd[1] * dist * frac + perp[1] * lat)
            if not (ARENA[0] + 1 < cx < ARENA[1] - 1 and
                    ARENA[0] + 1 < cy < ARENA[1] - 1):
                continue
            if not _safe_obs(cx, cy, r, [start, goal], gap=0.9):
                continue
            if not _min_sep(st_centers + centers, (cx, cy),
                            st_radii + radii, r):
                continue
            appear = int(rng.integers(max(20, max_step // 4),
                                      max(21, max_step // 2)))
            dyn_obs.append(DynamicObstacle(cx=cx, cy=cy, r=r,
                                           appear_step=appear))
            centers.append((cx, cy))
            radii.append(r)
            break

    return dyn_obs


def build_obstacles_narrow_corridor(start, goal, rng) -> List[Tuple]:
    """Four obstacles that funnel the drone through a narrow corridor."""
    mid  = (start[:2] + goal[:2]) / 2.0
    perp = np.array([-(goal[1] - start[1]), goal[0] - start[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    r    = 0.7
    gh   = 0.8 + float(rng.uniform(0.0, 0.4))
    foff = float(rng.uniform(-1.5, 1.5))
    fwd  = goal[:2] - start[:2]
    fwd /= np.linalg.norm(fwd) + 1e-9
    obs  = []
    for c in (mid + perp * (gh + r) + fwd * foff,
              mid - perp * (gh + r) + fwd * foff,
              mid + perp * (gh + r) - fwd * foff,
              mid - perp * (gh + r) - fwd * foff):
        if ARENA[0] + 1 < c[0] < ARENA[1] - 1 and ARENA[0] + 1 < c[1] < ARENA[1] - 1:
            obs.append(((float(c[0]), float(c[1]), 0.0), r))
    while len(obs) < 4:
        obs += build_static_circles(1, start, goal, rng)
    return obs[:4]


def build_obstacles_gap(start, goal, rng) -> List[Tuple]:
    """Four obstacles arranged around a single navigable gap."""
    mid  = (start[:2] + goal[:2]) / 2.0
    perp = np.array([-(goal[1] - start[1]), goal[0] - start[0]])
    perp /= np.linalg.norm(perp) + 1e-9
    rl   = 0.9
    gp   = float(rng.uniform(-0.5, 0.5))
    fwd  = goal[:2] - start[:2]
    fwd /= np.linalg.norm(fwd) + 1e-9
    obs  = []
    for c in (mid + perp * (gp + 2.1 * rl),
              mid + perp * (gp - 2.1 * rl),
              mid + perp * (gp + 5.0 * rl),
              mid + fwd * 2.0 + perp * float(rng.uniform(-1.5, 1.5))):
        if ARENA[0] + 1 < c[0] < ARENA[1] - 1 and ARENA[0] + 1 < c[1] < ARENA[1] - 1:
            obs.append(((float(c[0]), float(c[1]), 0.0), rl))
    while len(obs) < 4:
        obs += build_static_circles(1, start, goal, rng)
    return obs[:4]




def sample_start_goal(scenario: str, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a start and goal position appropriate for the scenario.
    TC-10 uses a fixed corridor so the obstacle layout is repeatable.
    """
    lo, hi = ARENA
    if scenario == "dynamic_obstacles":
        # Fixed corridor — dynamic obstacles are placed relative to this axis
        return (np.array([-7.0, 0.0, HOVER_Z]),
                np.array([ 7.0, 0.0, HOVER_Z]))
    # General: random start, goal at least 3 m apart
    start = np.array([float(rng.uniform(lo, hi)),
                      float(rng.uniform(lo, hi)), HOVER_Z])
    for _ in range(200):
        goal = np.array([float(rng.uniform(lo, hi)),
                         float(rng.uniform(lo, hi)), HOVER_Z])
        if np.linalg.norm(goal[:2] - start[:2]) >= 3.0:
            return start, goal
    return start, start + np.array([6.0, 0.0, 0.0])



def rk4_step(dyn, t: float, x, u, dt: float = DT):
    def f(tt, xx):
        return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t,          x)
    k2 = f(t + .5*dt,  x + .5*dt * k1)
    k3 = f(t + .5*dt,  x + .5*dt * k2)
    k4 = f(t +    dt,  x +    dt * k3)
    xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q = xn[6:10]
    xn[6:10] = q / (np.linalg.norm(q) + 1e-12)
    return xn



def compute_settling_time(dists: list, threshold: float, dt: float = DT) -> float:
    """First step at which distance stays below `threshold` for the rest of
    the episode.  Returns -1.0 if never achieved."""
    for i in range(len(dists)):
        if dists[i] < threshold and all(d < threshold for d in dists[i:]):
            return i * dt
    return -1.0

def compute_overshoot(dists: list, goal_dist_init: float) -> float:
    """Percentage overshoot after first minimum distance."""
    if not dists:
        return 0.0
    mi = int(np.argmin(dists))
    if mi >= len(dists) - 1:
        return 0.0
    return (max(dists[mi:]) - dists[mi]) / max(goal_dist_init, 1e-6) * 100.0

def compute_steady_state_error(dists: list, dt: float = DT,
                                window_s: float = 1.0) -> float:
    nw = int(window_s / dt)
    return (float(np.mean(np.abs(dists[-nw:])))
            if len(dists) >= nw else float(np.mean(np.abs(dists))))

def compute_action_smoothness(actions: list) -> float:
    if len(actions) < 2:
        return 0.0
    arr = np.array(actions)
    return float(np.mean(np.sum(np.diff(arr, axis=0) ** 2, axis=1)))

def compute_subgoal_consistency(subgoals: list) -> float:
    if len(subgoals) < 2:
        return 1.0
    cs = []
    for i in range(1, len(subgoals)):
        a, b = subgoals[i-1], subgoals[i]
        cs.append(float(np.dot(a, b) /
                        ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8))))
    return float(np.mean(cs))



@dataclass
class TrialResult:
    scenario:           str
    trial_idx:          int
    seed:               int
    n_static_obs:       int
    n_dynamic_obs:      int
    goal_dist_init:     float
    # Outcomes
    success:            bool
    collision:          bool
    timeout:            bool
    dynamic_obs_hit:    bool    
    # Position error
    final_dist:         float
    min_dist:           float
    mean_dist:          float
    # Integral error metrics
    rmse:               float
    iae:                float
    ise:                float
    itae:               float
    # Navigation quality
    path_length:        float
    path_length_ratio:  float
    near_miss_count:    int
    completion_time_s:  float
    steps_taken:        int
    # Control quality
    convergence_step:   int
    settling_time_s:    float
    overshoot:          float
    steady_state_error: float
    action_smoothness:  float
    subgoal_consistency: float
    # Thrust / altitude
    fz_mean:            float
    fz_max:             float
    fz_drift:           float
    z_drift:            float
    # Clearance
    min_obs_clearance:  float
    # dynamic obstacle metrics
    first_appear_step:  int     # earliest appear_step among dynamic obstacles; -1 if none
    reaction_step:      int    
    dyn_obs_near_miss_count: int
    dyn_obs_min_clearance:   float
    # Timing
    wall_time_s:        float


@dataclass
class ScenarioStats:
    scenario:               str
    n_trials:               int
    n_static_obs:           int
    n_dynamic_obs:          int
    # Outcomes
    success_rate:           float
    collision_rate:         float
    timeout_rate:           float
    dynamic_hit_rate:       float
    # Error
    mean_rmse:              float
    std_rmse:               float
    mean_iae:               float
    mean_ise:               float
    mean_itae:              float
    # Navigation
    mean_path_length:       float
    std_path_length:        float
    mean_plr:               float
    mean_near_miss:         float
    mean_completion_time:   float
    # Control quality
    mean_settling_time:     float
    mean_overshoot:         float
    mean_ss_error:          float
    mean_action_smooth:     float
    mean_subgoal_consist:   float
    # Position
    mean_final_dist:        float
    std_final_dist:         float
    mean_min_dist:          float
    # Thrust
    mean_fz_drift:          float
    std_fz_drift:           float
    mean_z_drift:           float
    # Clearance
    mean_obs_clearance:     float
    std_obs_clearance:      float
    mean_reaction_step:     float
    std_reaction_step:      float
    mean_dyn_obs_near_miss: float
    mean_dyn_obs_min_clearance: float
    dynamic_hit_by_dyn_rate: float
    # Timing
    mean_wall_time_s:       float



def run_trial(
        scenario:        str,
        trial_idx:       int,
        seed:            int,
        verbose:         bool = False,
        save_timeseries: bool = False,
        ts_dir:          str  = "benchmark_results/timeseries",
) -> TrialResult:
    t0  = time.perf_counter()
    rng = np.random.default_rng(seed)

    #Dynamics & controllers 
    params   = dynamics.QuadrotorParams()
    dyn      = dynamics.QuadrotorDynamics(params)

    ctrl = UHRCController(MODEL_PATH, STATS_PATH, device="cpu")
    ctrl.reset()

    # Episode geometry 
    start, goal    = sample_start_goal(scenario, rng)
    goal_dist_init = float(np.linalg.norm(goal[:2] - start[:2]))

    # Obstacle layout 
    dyn_obs_list: List[DynamicObstacle] = []

    if scenario == "dynamic_obstacles":
        static_circles = build_static_circles(DYN_OBS_N_STATIC, start, goal, rng)
        # Estimated flight steps at ~2 m/s cruise; used to bound appear windows
        est_flight_steps = int(goal_dist_init / (2.0 * DT))
        dyn_obs_list = build_dynamic_pop_obstacles(
            DYN_OBS_N_DYNAMIC, start, goal, rng,
            static_circles, est_flight_steps)
    elif scenario == "narrow_corridor":
        static_circles = build_obstacles_narrow_corridor(start, goal, rng)
    elif scenario == "gap_navigation":
        static_circles = build_obstacles_gap(start, goal, rng)
    else:
        # open_field — n=0, returns []
        static_circles = build_static_circles(
            ACTIVE_SCENARIOS.get(scenario, 0), start, goal, rng)

    n_static  = len(static_circles)
    n_dynamic = len(dyn_obs_list)

    first_appear_step = (min(d.appear_step for d in dyn_obs_list)
                         if dyn_obs_list else -1)

    # Initial state 
    x_curr = dyn.pack_state(
        start, np.zeros(3),
        np.array([1., 0., 0., 0.]),
        np.zeros(3), np.zeros(4))

    # Buffers 
    positions            = [start.copy()]
    z_log                = [float(start[2])]
    dists:               List[float] = []
    actions:             List[np.ndarray] = []   # [Fz, τx, τy, τz] each step
    subgoals:            List[np.ndarray] = []   # [vx_ref, vy_ref] from H-level
    obs_clearances:      List[float] = []
    dyn_clearances:      List[float] = []
    heading_log:         List[float] = []
    fz_log:              List[float] = []

    near_miss_count     = 0
    dyn_near_miss_count = 0
    appeared_already    = False
    reaction_step       = -1
    t_sim               = 0.0
    success             = False
    collision           = False
    dyn_hit             = False
    conv_step           = -1

    # Per-step dynamic obstacle positions for time-series saving
    dyn_pos_log: List[List[float]] = []

    for step in range(MAX_STEPS):
        r_I, v_I, q_BI, w_B, Omega = dyn.unpack_state(x_curr)
        psi = float(quat_euler.euler_from_q(q_BI)[2])
        heading_log.append(psi)

        # Active dynamic obstacles this step 
        active_tuples = [obs.as_circle_tuple()
                         for obs in dyn_obs_list if obs.is_active(step)]
        all_circles   = static_circles + active_tuples

        # Save flat xy positions for the time-series archive
        if dyn_obs_list:
            frame_xy: List[float] = []
            for obs in dyn_obs_list:
                frame_xy.extend([obs.cx, obs.cy])
            dyn_pos_log.append(frame_xy)

        # Reaction detection (heading change after first appearance) 
        any_new = any(obs.appear_step == step for obs in dyn_obs_list)
        if any_new and not appeared_already:
            appeared_already = True

        if appeared_already and reaction_step == -1 and len(heading_log) >= 2:
            dh = abs(heading_log[-1] - heading_log[-2])
            dh = min(dh, 2 * math.pi - dh)
            if dh > 0.02:
                reaction_step = step

        #  Lidar 
        lidar = get_lidar_scan(r_I, psi, all_circles, num_rays=32,
                               fov=math.pi, max_range=LIDAR_MAX)

        # UHRC inference: action = body wrench [Fz, τx, τy, τz] 

        u, sub_nn = ctrl.get_action(r_I, v_I, q_BI, w_B, lidar, goal)

        # Logging 
        fz_log.append(float(u[0]))
        actions.append(np.array(u, dtype=np.float32))
        subgoals.append(sub_nn.copy() if sub_nn is not None else np.zeros(2))

        dist = float(np.linalg.norm(r_I[:2] - goal[:2]))
        dists.append(dist)
        if conv_step == -1 and dist < CONVERGE_THRESH:
            conv_step = step

        # Clearance against all currently-visible obstacles
        if all_circles:
            cl = [float(np.linalg.norm(r_I[:2] - np.array([c[0], c[1]])) - r)
                  for (c, r) in all_circles]
            min_cl = min(cl)
            obs_clearances.append(min_cl)
            if min_cl < NEAR_MISS_THRESH:
                near_miss_count += 1

        # Clearance specifically against active dynamic obstacles (TC-10)
        if active_tuples:
            dcl = [float(np.linalg.norm(r_I[:2] - np.array([c[0], c[1]])) - r)
                   for (c, r) in active_tuples]
            min_dcl = min(dcl)
            dyn_clearances.append(min_dcl)
            if min_dcl < NEAR_MISS_THRESH:
                dyn_near_miss_count += 1

        if verbose and step % 100 == 0:
            n_act = sum(1 for obs in dyn_obs_list if obs.is_active(step))
            print(f"  [{scenario}] t{trial_idx} s{step:4d}"
                  f"  pos=({r_I[0]:6.2f},{r_I[1]:6.2f})"
                  f"  dist={dist:.2f}m"
                  f"  Fz={u[0]:.2f}"
                  f"  τ=({u[1]:+.3f},{u[2]:+.3f},{u[3]:+.3f})"
                  f"  active_dyn={n_act}")

        # Propagate 
        x_curr = rk4_step(dyn, t_sim, x_curr, u)
        t_sim += DT
        r_new, *_ = dyn.unpack_state(x_curr)
        positions.append(r_new.copy())
        z_log.append(float(r_new[2]))

        # Collision & success checks 
        hit_static = any(
            float(np.linalg.norm(r_new[:2] - np.array([c[0], c[1]]))) < float(r)
            for (c, r) in static_circles)
        hit_dyn = any(
            obs.is_active(step + 1) and
            float(np.linalg.norm(r_new[:2] - np.array([obs.cx, obs.cy]))) < obs.r
            for obs in dyn_obs_list)
        reached = float(np.linalg.norm(r_new[:2] - goal[:2])) < GOAL_RADIUS

        if hit_static or hit_dyn:
            collision = True
            dyn_hit   = hit_dyn
            break
        if reached:
            success = True
            break

    # Post-hoc metrics 
    steps_taken  = len(positions) - 1
    path         = np.array(positions)
    path_length  = float(np.sum(
        np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))

    dist_arr  = np.array(dists, dtype=float)
    t_arr     = np.arange(len(dist_arr)) * DT
    rmse      = float(np.sqrt(np.mean(dist_arr ** 2))) if len(dist_arr) else 0.0
    iae       = float(np.sum(np.abs(dist_arr))  * DT)  if len(dist_arr) else 0.0
    ise       = float(np.sum(dist_arr ** 2)     * DT)  if len(dist_arr) else 0.0
    itae      = float(np.sum(t_arr * np.abs(dist_arr)) * DT) if len(dist_arr) else 0.0

    final_dist  = float(np.linalg.norm(path[-1, :2] - goal[:2]))
    min_dist    = float(np.min(dist_arr))  if len(dist_arr) else goal_dist_init
    mean_dist   = float(np.mean(dist_arr)) if len(dist_arr) else goal_dist_init
    plr         = path_length / max(goal_dist_init, 1e-6)

    settling  = compute_settling_time(dists, CONVERGE_THRESH)
    overshoot = compute_overshoot(dists, goal_dist_init)
    ss_error  = compute_steady_state_error(dists) if success else -1.0
    smoothness = compute_action_smoothness(actions)
    sg_consist = compute_subgoal_consistency(subgoals)

    fz_arr   = np.array(fz_log) if fz_log else np.full(1, params.mass * params.g)
    fz_mean  = float(fz_arr.mean())
    fz_max   = float(fz_arr.max())
    fz_drift = float(np.mean(np.abs(fz_arr - params.mass * params.g)))
    z_arr    = np.array(z_log)
    z_drift  = float(np.mean(np.abs(z_arr - HOVER_Z)))   # altitude tracking error

    min_obs_cl  = float(min(obs_clearances)) if obs_clearances else -1.0
    dyn_min_cl  = float(min(dyn_clearances)) if dyn_clearances else -1.0

    # time-series save 
    if save_timeseries:
        Path(ts_dir).mkdir(parents=True, exist_ok=True)
        circ_arr = (np.array([[c[0], c[1], r] for (c, r) in static_circles],
                              dtype=np.float32)
                    if static_circles else np.zeros((0, 3), dtype=np.float32))
        dyn_arr  = (np.array([[obs.cx, obs.cy, obs.r, obs.appear_step]
                               for obs in dyn_obs_list], dtype=np.float32)
                    if dyn_obs_list else np.zeros((0, 4), dtype=np.float32))
        T_saved  = min(len(dyn_pos_log), steps_taken)
        if dyn_obs_list and T_saved > 0:
            dyn_traj = (np.array(dyn_pos_log[:T_saved], dtype=np.float32)
                        .reshape(T_saved, n_dynamic, 2))
        else:
            dyn_traj = np.zeros((T_saved, 0, 2), dtype=np.float32)

        np.savez_compressed(
            os.path.join(ts_dir, f"{scenario}_t{trial_idx:03d}_s{seed}.npz"),
            trajectory   = path.astype(np.float32),
            start        = start.astype(np.float32),
            goal         = goal.astype(np.float32),
            circles      = circ_arr,
            dynamic_obs  = dyn_arr,
            dynamic_traj = dyn_traj,
            dist_log     = dist_arr.astype(np.float32),
            fz_log       = fz_arr.astype(np.float32),
            z_log        = z_arr.astype(np.float32),
            subgoal_log  = np.array(subgoals, dtype=np.float32),
            success      = np.array([success], dtype=bool),
            collision    = np.array([collision], dtype=bool),
            dyn_hit      = np.array([dyn_hit], dtype=bool),
        )

    return TrialResult(
        scenario=scenario, trial_idx=trial_idx, seed=seed,
        n_static_obs=n_static, n_dynamic_obs=n_dynamic,
        goal_dist_init=goal_dist_init,
        success=success, collision=collision,
        timeout=not success and not collision,
        dynamic_obs_hit=dyn_hit,
        final_dist=final_dist, min_dist=min_dist, mean_dist=mean_dist,
        rmse=rmse, iae=iae, ise=ise, itae=itae,
        path_length=path_length, path_length_ratio=plr,
        near_miss_count=near_miss_count,
        completion_time_s=steps_taken * DT,
        steps_taken=steps_taken,
        convergence_step=conv_step,
        settling_time_s=settling,
        overshoot=overshoot,
        steady_state_error=ss_error,
        action_smoothness=smoothness,
        subgoal_consistency=sg_consist,
        fz_mean=fz_mean, fz_max=fz_max, fz_drift=fz_drift,
        z_drift=z_drift,
        min_obs_clearance=min_obs_cl,
        first_appear_step=first_appear_step,
        reaction_step=reaction_step,
        dyn_obs_near_miss_count=dyn_near_miss_count,
        dyn_obs_min_clearance=dyn_min_cl,
        wall_time_s=time.perf_counter() - t0,
    )


#  AGGREGATE STATISTICS

def compute_stats(trials: List[TrialResult]) -> ScenarioStats:
    assert trials
    def _m(v): return float(np.mean(v)) if v else float("nan")
    def _s(v): return float(np.std(v))  if v else float("nan")

    clears   = [t.min_obs_clearance  for t in trials if t.min_obs_clearance  >= 0]
    settle   = [t.settling_time_s    for t in trials if t.settling_time_s    >= 0]
    ss_ok    = [t.steady_state_error for t in trials if t.steady_state_error >= 0]
    react    = [t.reaction_step      for t in trials if t.reaction_step      >= 0]
    dyn_cl   = [t.dyn_obs_min_clearance for t in trials if t.dyn_obs_min_clearance >= 0]

    is_dyn   = trials[0].scenario == "dynamic_obstacles"

    return ScenarioStats(
        scenario=trials[0].scenario,
        n_trials=len(trials),
        n_static_obs=trials[0].n_static_obs,
        n_dynamic_obs=trials[0].n_dynamic_obs,
        success_rate=_m([t.success   for t in trials]),
        collision_rate=_m([t.collision for t in trials]),
        timeout_rate=_m([t.timeout   for t in trials]),
        dynamic_hit_rate=_m([t.dynamic_obs_hit for t in trials]),
        mean_rmse=_m([t.rmse for t in trials]),
        std_rmse=_s([t.rmse  for t in trials]),
        mean_iae=_m([t.iae   for t in trials]),
        mean_ise=_m([t.ise   for t in trials]),
        mean_itae=_m([t.itae for t in trials]),
        mean_path_length=_m([t.path_length      for t in trials]),
        std_path_length=_s([t.path_length       for t in trials]),
        mean_plr=_m([t.path_length_ratio        for t in trials]),
        mean_near_miss=_m([t.near_miss_count    for t in trials]),
        mean_completion_time=_m([t.completion_time_s for t in trials]),
        mean_settling_time=_m(settle),
        mean_overshoot=_m([t.overshoot          for t in trials]),
        mean_ss_error=_m(ss_ok),
        mean_action_smooth=_m([t.action_smoothness   for t in trials]),
        mean_subgoal_consist=_m([t.subgoal_consistency for t in trials]),
        mean_final_dist=_m([t.final_dist for t in trials]),
        std_final_dist=_s([t.final_dist  for t in trials]),
        mean_min_dist=_m([t.min_dist    for t in trials]),
        mean_fz_drift=_m([t.fz_drift   for t in trials]),
        std_fz_drift=_s([t.fz_drift    for t in trials]),
        mean_z_drift=_m([t.z_drift     for t in trials]),
        mean_obs_clearance=_m(clears),
        std_obs_clearance=_s(clears),
        mean_reaction_step=_m(react)       if is_dyn else float("nan"),
        std_reaction_step=_s(react)        if is_dyn else float("nan"),
        mean_dyn_obs_near_miss=_m([t.dyn_obs_near_miss_count for t in trials]) if is_dyn else float("nan"),
        mean_dyn_obs_min_clearance=_m(dyn_cl) if is_dyn else float("nan"),
        dynamic_hit_by_dyn_rate=_m([t.dynamic_obs_hit for t in trials]) if is_dyn else float("nan"),
        mean_wall_time_s=_m([t.wall_time_s for t in trials]),
    )


#  OUTPUT

def print_summary_table(stats_list: List[ScenarioStats]):
    sep = "=" * 120
    print(f"\n{sep}\n  UHRC Benchmark Results\n{sep}")
    hdr = (f"  {'TC':6s}  {'Scenario':20s}  {'Succ%':6s}  {'Coll%':5s}"
           f"  {'RMSE':7s}  {'IAE':8s}  {'ITAE':9s}"
           f"  {'PLR':5s}  {'Settle':7s}  {'Smooth':7s}"
           f"  {'SubgCos':7s}  {'React':6s}  {'DynClr':7s}")
    print(hdr)
    print("-" * 120)
    for s in stats_list:
        tc  = _TC_IDS.get(s.scenario, "TC-??")
        st  = f"{s.mean_settling_time:.2f}" if not math.isnan(s.mean_settling_time) else "  N/A"
        rct = (f"{s.mean_reaction_step:.0f}" if not math.isnan(s.mean_reaction_step)
               else "  N/A")
        dc  = (f"{s.mean_dyn_obs_min_clearance:.3f}" if not math.isnan(s.mean_dyn_obs_min_clearance)
               else "  N/A")
        print(f"  {tc:6s}  {s.scenario:20s}  {s.success_rate*100:5.1f}%"
              f"  {s.collision_rate*100:4.1f}%"
              f"  {s.mean_rmse:6.3f}  {s.mean_iae:7.3f}  {s.mean_itae:8.3f}"
              f"  {s.mean_plr:5.2f}  {st:>7s}  {s.mean_action_smooth:6.4f}"
              f"  {s.mean_subgoal_consist:6.3f}  {rct:>6s}  {dc:>7s}")
    print(sep)


def save_results(trials: List[TrialResult],
                 stats:  List[ScenarioStats],
                 out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    td = [asdict(t) for t in trials]
    if td:
        with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(td[0].keys()))
            w.writeheader(); w.writerows(td)
        print(f"  Saved per-trial CSV  → {out_dir}/trials.csv")

    sd = [asdict(s) for s in stats]
    if sd:
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sd[0].keys()))
            w.writeheader(); w.writerows(sd)
        print(f"  Saved summary CSV    → {out_dir}/summary.csv")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"trials": td, "summary": sd}, f, indent=2,
                  default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    print(f"  Saved full JSON      → {out_dir}/results.json")


def plot_summary(stats_list: List[ScenarioStats], out_dir: str):
    scenarios = [s.scenario.replace("_", "\n") for s in stats_list]
    x = np.arange(len(scenarios))
    w = 0.55

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("UHRC Benchmark Summary", fontsize=13)

    # 1. Outcomes
    ax = axes[0, 0]
    succ = [s.success_rate   * 100 for s in stats_list]
    coll = [s.collision_rate * 100 for s in stats_list]
    tout = [s.timeout_rate   * 100 for s in stats_list]
    ax.bar(x, succ, w, label="Success",   color="#4CAF50")
    ax.bar(x, coll, w, bottom=succ,       label="Collision", color="#F44336")
    ax.bar(x, tout, w,
           bottom=[a + b for a, b in zip(succ, coll)],
           label="Timeout", color="#FFC107")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("%"); ax.set_title("Outcomes"); ax.legend(fontsize=7)

    # 2. RMSE
    ax = axes[0, 1]
    vals = [s.mean_rmse for s in stats_list]
    errs = [s.std_rmse  for s in stats_list]
    ax.bar(x, vals, w, color="#2196F3", alpha=0.85)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE")

    # 3. ITAE
    ax = axes[0, 2]
    ax.bar(x, [s.mean_itae for s in stats_list], w, color="#9C27B0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("ITAE"); ax.set_title("ITAE")

    # 4. Action smoothness
    ax = axes[0, 3]
    ax.bar(x, [s.mean_action_smooth for s in stats_list], w,
           color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("||Δu||²"); ax.set_title("Action smoothness")

    # 5. Path length ratio
    ax = axes[1, 0]
    vals = [s.mean_plr for s in stats_list]
    errs = [s.std_path_length / max(s.mean_path_length, 1e-6) for s in stats_list]
    ax.bar(x, vals, w, color="#00BCD4", alpha=0.85)
    ax.axhline(1.0, color="green", ls="--", lw=0.8, label="Ideal PLR=1")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("PLR"); ax.set_title("Path Length Ratio"); ax.legend(fontsize=7)

    # 6. Altitude drift
    ax = axes[1, 1]
    ax.bar(x, [s.mean_z_drift for s in stats_list], w, color="#795548", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Mean |z − z_hover| (m)"); ax.set_title("Altitude drift")

    # 7. TC-10 reaction step
    ax = axes[1, 2]
    rct_vals = [s.mean_reaction_step if not math.isnan(s.mean_reaction_step)
                else 0.0 for s in stats_list]
    rct_cols = ["#FF5722" if not math.isnan(s.mean_reaction_step)
                else "#BDBDBD" for s in stats_list]
    ax.bar(x, rct_vals, w, color=rct_cols, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Steps"); ax.set_title("TC-10: Reaction step")

    # 8. TC-10 dynamic clearance
    ax = axes[1, 3]
    dc_vals = [s.mean_dyn_obs_min_clearance if not math.isnan(s.mean_dyn_obs_min_clearance)
               else 0.0 for s in stats_list]
    dc_cols = ["#FF5722" if not math.isnan(s.mean_dyn_obs_min_clearance)
               else "#BDBDBD" for s in stats_list]
    ax.bar(x, dc_vals, w, color=dc_cols, alpha=0.85)
    ax.axhline(NEAR_MISS_THRESH, color="red", ls="--", lw=0.8,
               label=f"Near-miss ({NEAR_MISS_THRESH} m)")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Min clearance (m)"); ax.set_title("TC-10: Dyn obs clearance")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "benchmark_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary figure → {fig_path}")



def parse_args():
    p = argparse.ArgumentParser(
        description="UHRC structured Monte-Carlo benchmark (ECNG 3020)")
    p.add_argument("--scenario",        type=str, default=None,
                   help=f"One of: {list(ACTIVE_SCENARIOS)}")
    p.add_argument("--n",               type=int, default=DEFAULT_TRIALS,
                   help="Number of Monte-Carlo trials per scenario")
    p.add_argument("--seed_base",       type=int, default=0)
    p.add_argument("--out",             type=str, default="benchmark_results")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--quick",           action="store_true",
                   help="Run first 2 scenarios with 5 trials each (smoke test)")
    p.add_argument("--save_timeseries", action="store_true",
                   help="Save per-trial .npz time-series to <out>/trials/")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        scenarios = list(ACTIVE_SCENARIOS)[:2]
        n_trials  = 5
    elif args.scenario:
        if args.scenario not in ACTIVE_SCENARIOS:
            print(f"Unknown scenario '{args.scenario}'."
                  f"  Choose from: {list(ACTIVE_SCENARIOS)}")
            return
        scenarios = [args.scenario]
        n_trials  = args.n
    else:
        scenarios = list(ACTIVE_SCENARIOS)
        n_trials  = args.n

    ts_dir = os.path.join(args.out, "trials")

    print(f"\n{'='*72}")
    print(f"  UHRC Benchmark  |  {len(scenarios)} scenario(s)  |"
          f"  {n_trials} trials each")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Stats : {STATS_PATH}")
    print(f"{'='*72}\n")

    all_trials: List[TrialResult]  = []
    all_stats:  List[ScenarioStats] = []
    done = 0

    for sc in scenarios:
        print(f"  {_TC_IDS.get(sc, '??')}  {sc}"
              f"  (static_obs={ACTIVE_SCENARIOS[sc]},"
              f"  dyn_obs={DYN_OBS_N_DYNAMIC if sc=='dynamic_obstacles' else 0},"
              f"  n={n_trials})")
        sc_trials: List[TrialResult] = []

        for i in range(n_trials):
            tr = run_trial(
                sc, i, args.seed_base + done,
                verbose=args.verbose,
                save_timeseries=args.save_timeseries,
                ts_dir=ts_dir,
            )
            sc_trials.append(tr)
            done += 1

            extra = ""
            if sc == "dynamic_obstacles":
                rct = (f"react@{tr.reaction_step}"
                       if tr.reaction_step >= 0 else "no-react")
                dcl = (f"{tr.dyn_obs_min_clearance:.2f}m"
                       if tr.dyn_obs_min_clearance >= 0 else "N/A")
                extra = f"  {rct}  dynClr={dcl}"
            print(
                  f"  RMSE={tr.rmse:.3f}  IAE={tr.iae:.2f}"
                  f"  PLR={tr.path_length_ratio:.2f}"
                  f"  z_drift={tr.z_drift:.3f}m{extra}", flush=True)

        stats = compute_stats(sc_trials)
        all_trials.extend(sc_trials)
        all_stats.append(stats)

        extra_summary = ""
        if sc == "dynamic_obstacles":
            rct = (f"{stats.mean_reaction_step:.0f}" if not math.isnan(
                stats.mean_reaction_step) else "N/A")
            dc  = (f"{stats.mean_dyn_obs_min_clearance:.3f}" if not math.isnan(
                stats.mean_dyn_obs_min_clearance) else "N/A")
            extra_summary = (f"  react={rct}steps"
                             f"  dynNM={stats.mean_dyn_obs_near_miss:.1f}"
                             f"  dynClr={dc}m")
        print(f"    → succ={stats.success_rate*100:.1f}%"
              f"  coll={stats.collision_rate*100:.1f}%"
              f"  RMSE={stats.mean_rmse:.3f}"
              f"  ITAE={stats.mean_itae:.3f}"
              f"  z_drift={stats.mean_z_drift:.3f}m"
              f"{extra_summary}\n")

    if all_stats:
        print_summary_table(all_stats)
    save_results(all_trials, all_stats, args.out)
    if all_stats:
        plot_summary(all_stats, args.out)

    if all_trials:
        ns = sum(t.success for t in all_trials)
        print(f"\n  Overall: {ns}/{len(all_trials)}"
              f"  ({ns/len(all_trials)*100:.1f}%) successful.\n")


if __name__ == "__main__":
    main()