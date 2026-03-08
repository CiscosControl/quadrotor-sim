# ppo_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from paramaters import QuadParams
from linear_model import linear_matrices


class DroneFormationEnv(gym.Env):
    """
    Full PPO environment for 3 linked drones.

    Each drone has 12 states and 4 inputs.
    Total state dimension = 36
    Total action dimension = 12

    Per-drone state:
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Modes:
        - "leader_only"
        - "full_formation"

    Reference types:
        - "hover"
        - "line"
        - "sinusoid"
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            dt=0.02,
            episode_steps=1000,
            mode="leader_only",
            reference_type="hover",
            w_leader_track=1.0,
            w_follower_form=0.2,
            w_vel=0.3,
            w_control=0.3,
            w_attitude=0.2,
            max_state_norm=500.0,
            noise_std=0.0,
    ):
    # def __init__(
    #     self,
    #     dt=0.02,
    #     episode_steps=1000,
    #     mode="leader_only",
    #     reference_type="hover",
    #     w_leader_track=4.0,
    #     w_follower_form=2.5,
    #     w_vel=1.0,
    #     w_control=0.02,
    #     w_attitude=0.2,
    #     max_state_norm=500.0,
    #     noise_std=0.0,
    # ):
        super().__init__()

        self.dt = dt
        self.episode_steps = episode_steps
        self.mode = mode
        self.reference_type = reference_type

        self.w_leader_track = w_leader_track
        self.w_follower_form = w_follower_form
        self.w_vel = w_vel
        self.w_control = w_control
        self.w_attitude = w_attitude

        self.max_state_norm = max_state_norm
        self.noise_std = noise_std

        self.step_count = 0

        # ------------------------------------------------------------------
        # Physical parameters and single-drone linear model
        # ------------------------------------------------------------------
        self.params = QuadParams()
        self.A, self.B = linear_matrices(self.params)

        self.n_drones = 3
        self.nx_single = 12
        self.nu_single = 4

        self.nx = self.n_drones * self.nx_single   # 36
        self.nu = self.n_drones * self.nu_single   # 12

        # Full state + leader error + follower1 relative error + follower2 relative error
        self.obs_dim = 36 + 12 + 12 + 12  # = 72

        # ------------------------------------------------------------------
        # Action bounds
        # [thrust_dev, tau_x, tau_y, tau_z] for each drone
        # ------------------------------------------------------------------

        # self.action_low = np.array(
        #     [-10.0, -2.0, -2.0, -2.0] * self.n_drones, dtype=np.float32
        # )
        # self.action_high = np.array(
        #     [10.0,  2.0,  2.0,  2.0] * self.n_drones, dtype=np.float32
        # )

        # ------------------------------------------------------------------
        # Physical action bounds
        # [thrust_dev, tau_x, tau_y, tau_z] for each drone
        # PPO will output normalized actions in [-1, 1], then we rescale.
        # ------------------------------------------------------------------
        self.physical_action_low = np.array(
            [-1.0, -0.2, -0.2, -0.2] * self.n_drones,
            dtype=np.float32
        )
        self.physical_action_high = np.array(
            [1.0, 0.2, 0.2, 0.2] * self.n_drones,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Desired follower offsets relative to leader
        # ------------------------------------------------------------------
        self.d1 = np.zeros(12)
        self.d1[0] = 2.0   # follower 1 offset in x

        self.d2 = np.zeros(12)
        self.d2[0] = -2.0  # follower 2 offset in x

        self.X = np.zeros(self.nx, dtype=np.float64)

    # ----------------------------------------------------------------------
    # Reference generator
    # ----------------------------------------------------------------------
    def leader_reference(self, t):
        xr = np.zeros(12)

        if self.reference_type == "hover":
            xr[0] = 0.0
            xr[1] = 0.0
            xr[2] = 2.0
            # velocities stay zero

        elif self.reference_type == "line":
            speed = 0.5
            xr[0] = speed * t
            xr[1] = 0.0
            xr[2] = 2.0

            xr[3] = speed
            xr[4] = 0.0
            xr[5] = 0.0

        elif self.reference_type == "sinusoid":
            omega = 0.1
            Ax = 10.0
            Ay = 5.0

            xr[0] = Ax * np.sin(omega * t)
            xr[1] = Ay * np.sin(2.0 * omega * t)
            xr[2] = 2.0

            xr[3] = Ax * omega * np.cos(omega * t)
            xr[4] = Ay * 2.0 * omega * np.cos(2.0 * omega * t)
            xr[5] = 0.0

        else:
            raise ValueError(f"Unknown reference_type: {self.reference_type}")

        return xr

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def split_state(self, X):
        x1 = X[0:12]
        x2 = X[12:24]
        x3 = X[24:36]
        return x1, x2, x3

    def split_action(self, U):
        u1 = U[0:4]
        u2 = U[4:8]
        u3 = U[8:12]
        return u1, u2, u3

    def rescale_action(self, action):
        """
        Map normalized action in [-1, 1] to physical control bounds.
        """
        action = np.clip(action, -1.0, 1.0)

        U = self.physical_action_low + 0.5 * (action + 1.0) * (
                self.physical_action_high - self.physical_action_low
        )
        return U

    def build_observation(self):
        x1, x2, x3 = self.split_state(self.X)
        t = self.step_count * self.dt
        xr1 = self.leader_reference(t)

        leader_err = x1 - xr1
        follower1_err = (x2 - x1) - self.d1
        follower2_err = (x3 - x1) - self.d2

        obs = np.concatenate([
            self.X,
            leader_err,
            follower1_err,
            follower2_err
        ]).astype(np.float32)

        return obs

    def dynamics(self, X, U, t):
        x1, x2, x3 = self.split_state(X)
        u1, u2, u3 = self.split_action(U)

        if self.mode == "leader_only":
            u2 = np.zeros(4)
            u3 = np.zeros(4)
        elif self.mode != "full_formation":
            raise ValueError(f"Unknown mode: {self.mode}")

        dx1 = self.A @ x1 + self.B @ u1
        dx2 = self.A @ x2 + self.B @ u2
        dx3 = self.A @ x3 + self.B @ u3

        dX = np.concatenate([dx1, dx2, dx3])

        if self.noise_std > 0.0:
            dX = dX + self.noise_std * np.random.randn(self.nx)

        return dX

    def calculate_reward(self, U):
        x1, x2, x3 = self.split_state(self.X)
        t = self.step_count * self.dt
        xr1 = self.leader_reference(t)

        leader_err = x1 - xr1
        follower1_err = (x2 - x1) - self.d1
        follower2_err = (x3 - x1) - self.d2

        # Leader tracking terms
        leader_pos_cost = np.linalg.norm(leader_err[0:3]) ** 2
        leader_vel_cost = np.linalg.norm(leader_err[3:6]) ** 2

        # Formation terms
        f1_pos_cost = np.linalg.norm(follower1_err[0:3]) ** 2
        f2_pos_cost = np.linalg.norm(follower2_err[0:3]) ** 2

        f1_vel_cost = np.linalg.norm(follower1_err[3:6]) ** 2
        f2_vel_cost = np.linalg.norm(follower2_err[3:6]) ** 2

        # Attitude / angular stabilization
        x1_att_cost = np.linalg.norm(x1[6:12]) ** 2
        x2_att_cost = np.linalg.norm(x2[6:12]) ** 2
        x3_att_cost = np.linalg.norm(x3[6:12]) ** 2

        # Control effort
        if self.mode == "leader_only":
            control_cost = np.linalg.norm(U[0:4]) ** 2
            total_cost = (
                self.w_leader_track * leader_pos_cost +
                self.w_vel * leader_vel_cost +
                self.w_attitude * x1_att_cost +
                self.w_control * control_cost
            )
        else:
            control_cost = np.linalg.norm(U) ** 2
            att_cost = x1_att_cost + x2_att_cost + x3_att_cost

            total_cost = (
                self.w_leader_track * leader_pos_cost +
                self.w_vel * leader_vel_cost +
                self.w_follower_form * (f1_pos_cost + f2_pos_cost) +
                self.w_vel * (f1_vel_cost + f2_vel_cost) +
                self.w_attitude * att_cost +
                self.w_control * control_cost
            )

        return -total_cost

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.X = np.zeros(self.nx, dtype=np.float64)

        xr0 = self.leader_reference(0.0)
        self.X[0:12] = xr0 + 0.03 * np.random.randn(12)
        self.X[12:24] = xr0 + self.d1 + 0.03 * np.random.randn(12)
        self.X[24:36] = xr0 + self.d2 + 0.03 * np.random.randn(12)
        # # Leader starts near reference
        # self.X[0:12] = xr0 + 0.1 * np.random.randn(12)
        #
        # # Followers start near formation offsets
        # self.X[12:24] = xr0 + self.d1 + 0.1 * np.random.randn(12)
        # self.X[24:36] = xr0 + self.d2 + 0.1 * np.random.randn(12)

        obs = self.build_observation()
        info = {}

        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        U = self.rescale_action(action)

        t = self.step_count * self.dt

        # Simple Euler integration
        dX = self.dynamics(self.X, U, t)
        self.X = self.X + self.dt * dX

        self.step_count += 1

        reward = self.calculate_reward(U)
        obs = self.build_observation()

        terminated = False
        truncated = False
        info = {"termination_reason": None}

        # Blow-up protection
        if np.linalg.norm(self.X) > self.max_state_norm:
            terminated = True
            reward -= 1000.0
            info["termination_reason"] = "state_norm_exceeded"

        if self.step_count >= self.episode_steps:
            truncated = True
            info["termination_reason"] = "episode_limit"

        return obs, reward, terminated, truncated, info
