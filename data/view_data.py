import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/Astardataref.npz"

def view_dataset():
    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    
    obs = data["obs"]
    actions = data["actions"]
    subgoals = data["subgoals"]
    ep_ids = data["episode_id"]
    
    total_transitions = len(obs)
    unique_eps = np.unique(ep_ids)
    
    print(f"Total Transitions: {total_transitions:,}")
    print(f"Total Episodes: {len(unique_eps)}")
    
    # ── 1. Calculate Braking vs Cruising Imbalance ──
    # Assuming obs[12] is goal_dist_norm (0.0 to 1.0)
    # 0.1 normalized roughly equals 2.0 meters (if GOAL_RANGE is 20)
    near_goal_mask = obs[:, 12] < 0.1
    braking_steps = np.sum(near_goal_mask)
    cruising_steps = total_transitions - braking_steps
    
    print("\nDataset Balance:")
    print(f"  Cruising Steps (>2m) : {cruising_steps:,} ({cruising_steps/total_transitions*100:.1f}%)")
    print(f"  Braking Steps  (<2m) : {braking_steps:,} ({braking_steps/total_transitions*100:.1f}%)")
    
    # ── 2. Plotting ──
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("UHRC Dataset Analysis", fontsize=16)
    
    # Plot A: XY Trajectories of first 15 episodes
    ax = axs[0, 0]
    for ep in unique_eps[:15]:
        mask = (ep_ids == ep)
        ep_obs = obs[mask]
        # Integrating velocity (obs 0,1) roughly to show path shape (since raw XY isn't in obs)
        # Note: This is an approximation since we don't save raw X/Y in the npz
        vx, vy = ep_obs[:, 0], ep_obs[:, 1]
        x = np.cumsum(vx) * 0.01
        y = np.cumsum(vy) * 0.01
        ax.plot(x, y, alpha=0.7)
        ax.plot(x[-1], y[-1], 'ro', markersize=4) # End point
    ax.set_title("Approximate XY Trajectories (First 15 Eps)")
    ax.grid(True, alpha=0.3)
    
    # Plot B: Histogram of Pitch/Roll Commands
    ax = axs[0, 1]
    phi_deg = np.degrees(actions[:, 0])
    theta_deg = np.degrees(actions[:, 1])
    ax.hist(phi_deg, bins=100, alpha=0.5, label='Roll (phi)', color='purple')
    ax.hist(theta_deg, bins=100, alpha=0.5, label='Pitch (theta)', color='orange')
    ax.set_title("Distribution of Attitude Commands")
    ax.set_xlabel("Degrees")
    ax.set_yscale('log') # Log scale reveals the extreme imbalance
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot C & D: A single episode's profile
    ep_mask = (ep_ids == unique_eps[558])
    ep_dist = obs[ep_mask][:, 12] * 20.0 # Assuming 20m GOAL_RANGE
    ep_phi = np.degrees(actions[ep_mask][:, 0])
    ep_theta = np.degrees(actions[ep_mask][:, 1])
    time_arr = np.arange(len(ep_dist)) * 0.01
    
    ax = axs[1, 0]
    ax.plot(time_arr, ep_dist, 'b-', label='Distance to Goal (m)')
    ax.set_title("Episode 0: Distance over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Meters")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axs[1, 1]
    ax.plot(time_arr, ep_phi, 'purple', label='Roll (phi)')
    ax.plot(time_arr, ep_theta, 'orange', label='Pitch (theta)')
    ax.set_title("Episode 0: Attitude Commands over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_dataset()