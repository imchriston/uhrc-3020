"""
uhrc_ctrl_wp.py — UHRC Controller with Waypoint Output
═══════════════════════════════════════════════════════════════════════════════

"""
from __future__ import annotations

import torch
import numpy as np
from models.hrm.uhrc import UHRC, UHRC_Config, UHRCCarry
import utils.quat_euler as quat_euler

MAX_LIDAR_RANGE = 5.0
GOAL_RANGE      = 20.0

# ── Toggle this when retraining with SEQ_LEN >= 2 ────────────────────────
PERSISTENT_CARRY = True   # False = reset every step (matches SEQ_LEN=1 training)


def _load_checkpoint(model, path, device):
    raw_sd   = torch.load(path, map_location=device, weights_only=True)
    model_sd = model.state_dict()
    compatible   = {}
    size_skipped = []
    for k, v in raw_sd.items():
        if k not in model_sd:
            continue
        if v.shape != model_sd[k].shape:
            size_skipped.append(k)
            continue
        compatible[k] = v
    missing, _ = model.load_state_dict(compatible, strict=False)
    if size_skipped:
        print(f"⚠️  {len(size_skipped)} layer(s) skipped (shape mismatch)")
    if missing:
        print(f"⚠️  {len(missing)} layer(s) randomly initialised")
    if not size_skipped and not missing:
        print("✅  Checkpoint loaded cleanly.")


class UHRCController:
    """
    Inference wrapper.  Returns (action[4], waypoint[2]).

    Goal-reached logic:  When dist_to_goal < GOAL_TOL, the controller
    overrides to hover thrust and zero torques (same as training hover phase).
    """

    GOAL_TOL = 0.5   # must match training REACH_RADIUS

    def __init__(self, model_path: str, stats_path: str, device: str = "cpu"):
        self.device = device

        # Normalisation
        print(f"Loading stats  : {stats_path}")
        stats = np.load(stats_path)
        self.obs_mean = torch.from_numpy(stats["obs_mean"].astype(np.float32)).to(device)
        self.obs_std  = torch.clamp(
            torch.from_numpy(stats["obs_std"].astype(np.float32)).to(device), min=1e-3
        )

        # Verify obs dimension matches what we build (45)
        assert self.obs_mean.shape[0] == 45, (
            f"Norm stats have {self.obs_mean.shape[0]} dims — expected 45. "
            f"Check if stats were generated with Omega (49-dim) instead of without (45-dim)."
        )

        # Model
        print(f"Loading model  : {model_path}")
        config = UHRC_Config(
            hidden_size=256, carry_len=16, expansion=4.0, num_heads=4,
            H_cycles=2, L_cycles=2, H_layers=2, L_layers=2,
            hover_thrust=9.81, detach_carry=True,
        )
        self.model = UHRC(config).to(device)
        _load_checkpoint(self.model, model_path, device)
        self.model.eval()

        self.carry: UHRCCarry | None = None
        self._reached = False

    def reset(self):
        self.carry = None
        self._reached = False

    def get_action(
        self,
        r_I: np.ndarray,
        v_I: np.ndarray,
        q_BI: np.ndarray,
        w_B: np.ndarray,
        lidar: np.ndarray,
        goal_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            action   [4]  [Fz, τx, τy, τz]
        """
        goal_dist = float(np.linalg.norm(goal_pos[:2] - r_I[:2]))

        if goal_dist < self.GOAL_TOL:
            self._reached = True

        if self._reached:
            return np.array([9.81, 0.0, 0.0, 0.0], dtype=np.float64), \
                   np.zeros(2, dtype=np.float32)

        obs = self._build_obs(r_I, v_I, q_BI, w_B, lidar, goal_pos)
        return self._infer(obs)

    # ── Observation (45-dim, NO Omega) ────────────────────────────────────
    def _build_obs(self, r_I, v_I, q_BI, w_B, lidar, goal_pos):
        """
        Build 45-dim observation.  Layout MUST match generate_data.py build_obs():
          v_B(3) | w_B(3) | g_B(3) | goal_rel_B(3) | goal_dist_norm(1) | lidar(32)
        
        """
        R_BI = quat_euler.R_BI_from_q(q_BI).T
        v_B        = R_BI @ v_I
        g_B        = R_BI @ np.array([0.0, 0.0, -9.81])
        goal_rel_B = R_BI @ (goal_pos - r_I)

        goal_dist = float(np.linalg.norm(goal_rel_B)) + 1e-6
        goal_dist_norm = np.array(
            [min(goal_dist, GOAL_RANGE) / GOAL_RANGE], dtype=np.float32
        )
        if goal_dist > GOAL_RANGE:
            goal_rel_B = goal_rel_B / goal_dist * GOAL_RANGE

        lidar_norm = np.clip(lidar / MAX_LIDAR_RANGE, 0.0, 1.0)

        obs = np.concatenate([
            v_B.astype(np.float32),         # [0:3]
            w_B.astype(np.float32),         # [3:6]
            g_B.astype(np.float32),         # [6:9]
            goal_rel_B.astype(np.float32),  # [9:12]
            goal_dist_norm,                 # [12]
            lidar_norm.astype(np.float32),  # [13:45]
        ])
        assert obs.shape == (45,), f"Obs shape {obs.shape} != (45,)"
        return obs

    # ── Inference ─────────────────────────────────────────────────────────
    def _infer(self, obs):
        x_norm = (torch.from_numpy(obs).float().to(self.device)
                  - self.obs_mean) / self.obs_std
        x_input = x_norm.unsqueeze(0)   # [1, 45]

        # ── CARRY POLICY ──────────────────────────────────────────────────

        if not PERSISTENT_CARRY:
            self.carry = None

        with torch.no_grad():
            action_t, subgoal_t, self.carry = self.model(
                x_input, carry=self.carry
            )

        action  = action_t[0].cpu().numpy()
        subgoal = subgoal_t[0].cpu().numpy()

        waypoint = subgoal[:2].copy()

        # Safety clamp
        action[0] = float(np.clip(action[0],  4.0, 20.0))   # Fz
        action[1] = float(np.clip(action[1], -0.3,  0.3))   # τx
        action[2] = float(np.clip(action[2], -0.3,  0.3))   # τy
        action[3] = float(np.clip(action[3], -0.3,  0.3))   # τz
        return action, subgoal