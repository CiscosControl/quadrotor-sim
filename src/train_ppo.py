# train_ppo.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ppo_env import DroneFormationEnv


# ============================================================
# Environment factory
# ============================================================
def make_env(
    mode,
    reference_type,
    log_dir,
    dt=0.02,
    episode_steps=150,
    w_leader_track=4.0,
    w_follower_form=3.0,
    w_vel=0.8,
    w_control=0.01,
    noise_std=0.0,
):
    env = DroneFormationEnv(
        dt=dt,
        episode_steps=episode_steps,
        mode=mode,
        reference_type=reference_type,
        w_leader_track=w_leader_track,
        w_follower_form=w_follower_form,
        w_vel=w_vel,
        w_control=w_control,
        noise_std=noise_std,
    )

    env = Monitor(env, log_dir)
    return env


# ============================================================
# Rollout evaluation
# ============================================================
def evaluate_policy(model, env, n_steps=1000):
    obs, _ = env.reset()
    rewards = []
    states = []
    actions = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        states.append(env.X.copy())
        actions.append(action.copy())

        if terminated or truncated:
            break

    return np.array(rewards), np.array(states), np.array(actions)


# ============================================================
# Plotting helpers
# ============================================================
def plot_reward_curve(rewards, title_prefix="Final"):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title(f"{title_prefix} Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)


def plot_3d_trajectories(states, title_prefix="Final"):
    if len(states) == 0:
        print("No states to plot.")
        return

    x1 = states[:, 0:12]
    x2 = states[:, 12:24]
    x3 = states[:, 24:36]

    p1 = x1[:, 0:3]
    p2 = x2[:, 0:3]
    p3 = x3[:, 0:3]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], label="Leader")
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], label="Follower 1")
    ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], label="Follower 2")

    ax.set_title(f"{title_prefix} 3-Drone PPO Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()


def plot_xy_trajectories(states, title_prefix="Final"):
    if len(states) == 0:
        return

    x1 = states[:, 0:12]
    x2 = states[:, 12:24]
    x3 = states[:, 24:36]

    p1 = x1[:, 0:2]
    p2 = x2[:, 0:2]
    p3 = x3[:, 0:2]

    plt.figure(figsize=(10, 5))
    plt.plot(p1[:, 0], p1[:, 1], label="Leader")
    plt.plot(p2[:, 0], p2[:, 1], label="Follower 1")
    plt.plot(p3[:, 0], p3[:, 1], label="Follower 2")
    plt.title(f"{title_prefix} XY Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)


def plot_planar_interdrone_distances(states, title_prefix="Final"):
    if len(states) == 0:
        return

    x1 = states[:, 0:12]
    x2 = states[:, 12:24]
    x3 = states[:, 24:36]

    p1_xy = x1[:, 0:2]
    p2_xy = x2[:, 0:2]
    p3_xy = x3[:, 0:2]

    d12 = np.linalg.norm(p2_xy - p1_xy, axis=1)
    d13 = np.linalg.norm(p3_xy - p1_xy, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(d12, label="||p2_xy - p1_xy||")
    plt.plot(d13, label="||p3_xy - p1_xy||")
    plt.title(f"{title_prefix} Planar Inter-Drone Distances")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)


def plot_control_inputs(actions, title_prefix="Final"):
    if len(actions) == 0:
        return

    plt.figure(figsize=(11, 5))
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f"u{i+1}")

    plt.title(f"{title_prefix} PPO Actions")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=8)


def plot_all_results(rewards, states, actions, title_prefix="Final"):
    plot_reward_curve(rewards, title_prefix=title_prefix)
    plot_3d_trajectories(states, title_prefix=title_prefix)
    plot_xy_trajectories(states, title_prefix=title_prefix)
    plot_planar_interdrone_distances(states, title_prefix=title_prefix)
    plot_control_inputs(actions, title_prefix=title_prefix)


# ============================================================
# Animation
# ============================================================
def animate_results(states, env, title_prefix="Final"):
    if len(states) == 0:
        print("No states to animate.")
        return None

    x1 = states[:, 0:12]
    x2 = states[:, 12:24]
    x3 = states[:, 24:36]

    p1 = x1[:, 0:3]
    p2 = x2[:, 0:3]
    p3 = x3[:, 0:3]

    x1_traj, y1_traj = p1[:, 0], p1[:, 1]
    x2_traj, y2_traj = p2[:, 0], p2[:, 1]
    x3_traj, y3_traj = p3[:, 0], p3[:, 1]

    t_vals = np.arange(len(states)) * env.dt
    ref_points = np.array([env.leader_reference(t) for t in t_vals])
    ref_x = ref_points[:, 0]
    ref_y = ref_points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{title_prefix} XY Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)

    all_x = np.concatenate([x1_traj, x2_traj, x3_traj, ref_x])
    all_y = np.concatenate([y1_traj, y2_traj, y3_traj, ref_y])

    xpad = 0.5 + 0.1 * (all_x.max() - all_x.min() + 1e-6)
    ypad = 0.5 + 0.1 * (all_y.max() - all_y.min() + 1e-6)

    ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
    ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)

    ax.plot(ref_x, ref_y, "r--", alpha=0.3, label="Reference Path")

    line_l, = ax.plot([], [], "b-", alpha=0.30, label="Leader Trail")
    line_f1, = ax.plot([], [], "orange", alpha=0.30, label="Follower 1 Trail")
    line_f2, = ax.plot([], [], "g-", alpha=0.30, label="Follower 2 Trail")

    dot_l, = ax.plot([], [], "bo", markersize=8, label="Leader")
    dot_f1, = ax.plot([], [], "o", color="orange", markersize=7, label="Follower 1")
    dot_f2, = ax.plot([], [], "go", markersize=7, label="Follower 2")

    spring_l_f1, = ax.plot([], [], "k-", linewidth=1.2, alpha=0.7, label="Link L-F1")
    spring_l_f2, = ax.plot([], [], "k-", linewidth=1.0, alpha=0.5, label="Link L-F2")

    ax.legend(loc="upper right")

    def animate(i):
        line_l.set_data(x1_traj[:i + 1], y1_traj[:i + 1])
        line_f1.set_data(x2_traj[:i + 1], y2_traj[:i + 1])
        line_f2.set_data(x3_traj[:i + 1], y3_traj[:i + 1])

        dot_l.set_data([x1_traj[i]], [y1_traj[i]])
        dot_f1.set_data([x2_traj[i]], [y2_traj[i]])
        dot_f2.set_data([x3_traj[i]], [y3_traj[i]])

        spring_l_f1.set_data([x1_traj[i], x2_traj[i]], [y1_traj[i], y2_traj[i]])
        spring_l_f2.set_data([x1_traj[i], x3_traj[i]], [y1_traj[i], y3_traj[i]])

        return (
            line_l, line_f1, line_f2,
            dot_l, dot_f1, dot_f2,
            spring_l_f1, spring_l_f2
        )

    ani = FuncAnimation(
        fig,
        animate,
        frames=len(states),
        interval=40,
        blit=True,
        repeat=False
    )

    return ani


# ============================================================
# Curriculum training helper
# ============================================================
def run_stage(model, stage_name, mode, reference_type, total_timesteps, env_kwargs):
    print(f"\n===== Starting stage: {stage_name} =====")
    print(f"Mode: {mode}, Reference: {reference_type}, Timesteps: {total_timesteps}")

    log_dir = os.path.join("ppo_logs", stage_name)
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(
        mode=mode,
        reference_type=reference_type,
        log_dir=log_dir,
        **env_kwargs
    )

    model.set_env(env)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    return model


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs("ppo_logs", exist_ok=True)
    os.makedirs("ppo_models", exist_ok=True)
    os.makedirs("ppo_tensorboard", exist_ok=True)
    os.makedirs("ppo_media", exist_ok=True)

    # ------------------------------------------------------------
    # Shared environment settings
    # ------------------------------------------------------------
    env_kwargs = dict(
        dt=0.02,
        episode_steps=800,
        w_leader_track=5.0,
        w_follower_form = 0.6,
        w_vel = 0.2,
        w_control = 0.002,
        noise_std=0.0,
    )
    # env_kwargs = dict(
    #     dt=0.02,
    #     episode_steps=150,
    #     w_leader_track=4.0,
    #     w_follower_form=3.0,
    #     w_vel=0.8,
    #     w_control=0.01,
    #     noise_std=0.0,
    # )

    # ------------------------------------------------------------
    # Quick environment check
    # ------------------------------------------------------------
    test_env = DroneFormationEnv(
        mode="leader_only",
        reference_type="hover",
        **env_kwargs
    )
    check_env(test_env, warn=True)

    print("Observation dimension:", test_env.observation_space.shape[0])
    print("Action dimension:", test_env.action_space.shape[0])

    # ------------------------------------------------------------
    # Initial training environment
    # ------------------------------------------------------------
    init_env = make_env(
        mode="leader_only",
        reference_type="hover",
        log_dir=os.path.join("ppo_logs", "init"),
        **env_kwargs
    )

    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO(
        policy="MlpPolicy",
        env=init_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=init_env,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=256,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.005,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     tensorboard_log="./ppo_tensorboard/",
    # )

    # ------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------
    curriculum = [
        {
            "stage_name": "stage1_leader_hover",
            "mode": "leader_only",
            "reference_type": "hover",
            "total_timesteps": 300_000,
        },
        {
            "stage_name": "stage2_leader_line",
            "mode": "leader_only",
            "reference_type": "line",
            "total_timesteps": 400_000,
        },
        {
            "stage_name": "stage3_full_hover",
            "mode": "full_formation",
            "reference_type": "hover",
            "total_timesteps": 500_000,
        },
        {
            "stage_name": "stage4_full_line",
            "mode": "full_formation",
            "reference_type": "line",
            "total_timesteps": 700_000,
        },
        {
            "stage_name": "stage5_full_lissajous",
            "mode": "full_formation",
            "reference_type": "lissajous",
            "total_timesteps": 1_200_000,
        },
    ]

    for stage in curriculum:
        model = run_stage(
            model=model,
            stage_name=stage["stage_name"],
            mode=stage["mode"],
            reference_type=stage["reference_type"],
            total_timesteps=stage["total_timesteps"],
            env_kwargs=env_kwargs,
        )

        save_path = os.path.join("ppo_models", f'{stage["stage_name"]}.zip')
        model.save(save_path)
        print(f"Saved model to: {save_path}")

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------
    eval_env = DroneFormationEnv(
        mode="full_formation",
        reference_type="lissajous",
        **env_kwargs
    )

    rewards, states, actions = evaluate_policy(model, eval_env, n_steps=1000)

    print("\n===== Final Evaluation =====")
    print("Mean reward per step:", rewards.mean())
    print("Total reward:", rewards.sum())

    # create all figures first
    plot_all_results(rewards, states, actions, title_prefix="Final")

    # create animation and keep reference alive
    ani = animate_results(states, eval_env, title_prefix="Final")

    # optional: save animation as gif
    if ani is not None:
        gif_path = os.path.join("ppo_media", "final_xy_animation.gif")
        ani.save(gif_path, writer="pillow", fps=25)
        print(f"Saved animation to: {gif_path}")

    # show all figures together
    plt.show()


if __name__ == "__main__":
    main()