# train_ppo.py
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ppo_env import DroneFormationEnv


def make_env(
    mode,
    reference_type,
    log_dir,
    dt=0.02,
    episode_steps=1000,
    w_leader_track=4.0,
    w_follower_form=2.5,
    w_vel=1.0,
    w_control=0.02,
    w_attitude=0.2,
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
        w_attitude=w_attitude,
        noise_std=noise_std,
    )
    env = Monitor(env, log_dir)
    return env


def evaluate_policy(model, env, n_steps=1000):
    obs, _ = env.reset()
    rewards = []
    states = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        states.append(env.X.copy())

        if terminated or truncated:
            break

    return np.array(rewards), np.array(states)


def plot_results(states, title_prefix=""):
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
    plt.show()

    d12 = np.linalg.norm(p2 - p1, axis=1)
    d13 = np.linalg.norm(p3 - p1, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(d12, label="||p2 - p1||")
    plt.plot(d13, label="||p3 - p1||")
    plt.title(f"{title_prefix} Inter-Drone Distances")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()


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

    # Attach the new environment to the same model
    model.set_env(env)

    # Continue training without resetting timestep counter
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    return model


def main():
    os.makedirs("ppo_logs", exist_ok=True)
    os.makedirs("ppo_models", exist_ok=True)
    os.makedirs("ppo_tensorboard", exist_ok=True)

    # -------------------------------------------------------------
    # Shared environment settings
    # -------------------------------------------------------------
    env_kwargs = dict(
        dt=0.02,
        episode_steps=150,
        w_leader_track=1.0,
        w_follower_form=0.2,
        w_vel=0.3,
        w_control=0.3,
        w_attitude=0.2,
        noise_std=0.0,
    )
    # env_kwargs = dict(
    #     dt=0.02,
    #     episode_steps=1000,
    #     w_leader_track=6.0,
    #     w_follower_form=3.0,
    #     w_vel=1.5,
    #     w_control=0.03,
    #     w_attitude=0.3,
    #     noise_std=0.0,
    # )

    # -------------------------------------------------------------
    # Quick check on one simple env
    # -------------------------------------------------------------
    test_env = DroneFormationEnv(
        mode="leader_only",
        reference_type="hover",
        **env_kwargs
    )
    check_env(test_env, warn=True)

    # -------------------------------------------------------------
    # Initial environment and PPO model
    # -------------------------------------------------------------
    init_env = make_env(
        mode="leader_only",
        reference_type="hover",
        log_dir=os.path.join("ppo_logs", "init"),
        **env_kwargs
    )

    model = PPO(
        policy="MlpPolicy",
        env=init_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )

    # -------------------------------------------------------------
    # Curriculum stages
    # -------------------------------------------------------------
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
            "total_timesteps": 300_000,
        },
        {
            "stage_name": "stage3_full_hover",
            "mode": "full_formation",
            "reference_type": "hover",
            "total_timesteps": 300_000,
        },
    ]
    # curriculum = [
    #     {
    #         "stage_name": "stage1_leader_hover",
    #         "mode": "leader_only",
    #         "reference_type": "hover",
    #         "total_timesteps": 200_000,
    #     },
    #     {
    #         "stage_name": "stage2_leader_line",
    #         "mode": "leader_only",
    #         "reference_type": "line",
    #         "total_timesteps": 300_000,
    #     },
    #     {
    #         "stage_name": "stage3_full_hover",
    #         "mode": "full_formation",
    #         "reference_type": "hover",
    #         "total_timesteps": 300_000,
    #     },
    #     {
    #         "stage_name": "stage4_full_line",
    #         "mode": "full_formation",
    #         "reference_type": "line",
    #         "total_timesteps": 500_000,
    #     },
    #     {
    #         "stage_name": "stage5_full_sinusoid",
    #         "mode": "full_formation",
    #         "reference_type": "sinusoid",
    #         "total_timesteps": 700_000,
    #     },
    # ]

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

    # -------------------------------------------------------------
    # Final evaluation on hardest task
    # -------------------------------------------------------------
    eval_env = DroneFormationEnv(
        mode="full_formation",
        reference_type="hover",
        **env_kwargs
    )
    # eval_env = DroneFormationEnv(
    #     mode="full_formation",
    #     reference_type="sinusoid",
    #     **env_kwargs
    # )

    rewards, states = evaluate_policy(model, eval_env, n_steps=1000)

    print("\n===== Final Evaluation =====")
    print("Mean reward per step:", rewards.mean())
    print("Total reward:", rewards.sum())

    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Reward per Step (Final Evaluation)")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    plot_results(states, title_prefix="Final")


if __name__ == "__main__":
    main()