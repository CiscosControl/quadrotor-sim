# ppo_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from paramaters import QuadParams
from linear_model import linear_matrices


class DroneFormationEnv(gym.Env):
    """
    PPO environment for 3 linked drones with enforced hover dynamics.

    Internal state per drone remains 12D:
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Total internal state dimension = 36

    PPO action is reduced to planar commands only:
    [ax1, ay1, ax2, ay2, ax3, ay3]

    Total PPO action dimension = 6

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
        episode_steps=150,
        mode="leader_only",
        reference_type="hover",
        w_leader_track=4.0,
        w_follower_form=2.5,
        w_vel=1.0,
        w_control=0.02,
        max_state_norm=500.0,
        noise_std=0.0,
    ):
        super().__init__()

        self.dt = dt
        self.episode_steps = episode_steps
        self.mode = mode
        self.reference_type = reference_type

        self.w_leader_track = w_leader_track
        self.w_follower_form = w_follower_form
        self.w_vel = w_vel
        self.w_control = w_control

        self.max_state_norm = max_state_norm
        self.noise_std = noise_std

        self.step_count = 0

        # ------------------------------------------------------------------
        # Physical parameters and original single-drone linear model
        # kept for compatibility / reference
        # ------------------------------------------------------------------
        self.params = QuadParams()
        self.A, self.B = linear_matrices(self.params)

        self.n_drones = 3
        self.nx_single = 12
        self.nx = self.n_drones * self.nx_single  # 36

        # PPO controls only planar acceleration-like commands
        self.nu_rl_single = 2
        self.nu = self.n_drones * self.nu_rl_single  # 6

        # Observation contains only planar task-relevant errors:
        # leader_err[x, y, vx, vy]
        # follower1_rel_err[x, y, vx, vy]
        # follower2_rel_err[x, y, vx, vy]
        self.obs_dim = 12

        # ------------------------------------------------------------------
        # Action bounds
        # [ax_cmd, ay_cmd] for each drone
        # ------------------------------------------------------------------
        self.action_low = np.array([-2.0, -2.0] * self.n_drones, dtype=np.float32)
        self.action_high = np.array([2.0, 2.0] * self.n_drones, dtype=np.float32)
        # self.action_low = np.array([-2.0, -2.0] * self.n_drones, dtype=np.float32)
        # self.action_high = np.array([2.0, 2.0] * self.n_drones, dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
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
        self.d1 = np.zeros(12, dtype=np.float64)
        self.d1[0] = 2.0    # follower 1 offset in x

        self.d2 = np.zeros(12, dtype=np.float64)
        self.d2[0] = -2.0   # follower 2 offset in x

        self.X = np.zeros(self.nx, dtype=np.float64)

        # ------------------------------------------------------------------
        # Hover-lock / damping gains
        # ------------------------------------------------------------------
        self.kv_xy = 1.5

        self.kz = 4.0
        self.kvz = 2.5

        self.katt = 3.0
        self.krat = 1.5

        self.kyaw = 2.0
        self.kryaw = 1.0

    # ----------------------------------------------------------------------
    # Reference generator
    # ----------------------------------------------------------------------
    def leader_reference(self, t):
        xr = np.zeros(12, dtype=np.float64)

        if self.reference_type == "hover":
            xr[0] = 0.0
            xr[1] = 0.0
            xr[2] = 2.0

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
        elif self.reference_type == "lissajous":

            Ax = 2.5
            Ay = 2.0

            wx = 0.35
            wy = 0.7

            delta = np.pi / 2

            xr[0] = Ax * np.sin(wx * t + delta)
            xr[1] = Ay * np.sin(wy * t)

            xr[2] = 2.0

            xr[3] = Ax * wx * np.cos(wx * t + delta)
            xr[4] = Ay * wy * np.cos(wy * t)
            xr[5] = 0.0
        # elif self.reference_type == "lissajous":
        #     xr = np.zeros(12)
        #              Ax = 4.0
        #             Ay = 2.5
        #             wx = 0.4
        #             wy = 0.8
        #     A_x = 10.0
        #     A_y = 5.0
        #     omega = 0.2
        #
        #     # position reference
        #     xr[0] = A_x * np.sin(omega * t)  # x
        #     xr[1] = A_y * np.sin(2.0 * omega * t)  # y
        #     xr[2] = 2.0  # z fixed hover plane
        #
        #     # velocity reference
        #     xr[3] = A_x * omega * np.cos(omega * t)  # vx
        #     xr[4] = 2.0 * A_y * omega * np.cos(2.0 * omega * t)  # vy
        #     xr[5] = 0.0  # vz

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

    def split_planar_action(self, U):
        u1 = U[0:2]
        u2 = U[2:4]
        u3 = U[4:6]
        return u1, u2, u3

    def build_observation(self):
        x1, x2, x3 = self.split_state(self.X)
        t = self.step_count * self.dt
        xr1 = self.leader_reference(t)

        leader_err = x1 - xr1
        follower1_err = (x2 - x1) - self.d1
        follower2_err = (x3 - x1) - self.d2

        obs = np.concatenate([
            leader_err[[0, 1, 3, 4]],
            follower1_err[[0, 1, 3, 4]],
            follower2_err[[0, 1, 3, 4]],
        ]).astype(np.float32)

        return obs

    def drone_planar_hover_dynamics(self, x, u_xy, xr):
        """
        Enforced-hover drone dynamics:
        - PPO controls planar acceleration-like channels
        - z is regulated to hover altitude
        - attitude and angular rates decay to zero
        """
        dx = np.zeros(12, dtype=np.float64)

        ax_cmd, ay_cmd = u_xy

        # Position kinematics
        dx[0] = x[3]   # x_dot = vx
        dx[1] = x[4]   # y_dot = vy
        dx[2] = x[5]   # z_dot = vz

        # Planar dynamics controlled by PPO
        dx[3] = ax_cmd - self.kv_xy * x[3]
        dx[4] = ay_cmd - self.kv_xy * x[4]

        # Altitude hold
        z_ref = xr[2]
        dx[5] = -self.kz * (x[2] - z_ref) - self.kvz * x[5]

        # Angle kinematics
        dx[6] = x[9]    # phi_dot = p
        dx[7] = x[10]   # theta_dot = q
        dx[8] = x[11]   # psi_dot = r

        # Angle/rate stabilization
        dx[9] = -self.katt * x[6] - self.krat * x[9]
        dx[10] = -self.katt * x[7] - self.krat * x[10]
        dx[11] = -self.kyaw * x[8] - self.kryaw * x[11]

        return dx

    def dynamics(self, X, U, t):
        x1, x2, x3 = self.split_state(X)
        u1_xy, u2_xy, u3_xy = self.split_planar_action(U)

        xr1 = self.leader_reference(t)
        xr2 = xr1 + self.d1
        xr3 = xr1 + self.d2

        if self.mode == "leader_only":
            u2_xy = np.zeros(2, dtype=np.float64)
            u3_xy = np.zeros(2, dtype=np.float64)
        elif self.mode != "full_formation":
            raise ValueError(f"Unknown mode: {self.mode}")

        dx1 = self.drone_planar_hover_dynamics(x1, u1_xy, xr1)
        dx2 = self.drone_planar_hover_dynamics(x2, u2_xy, xr2)
        dx3 = self.drone_planar_hover_dynamics(x3, u3_xy, xr3)

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

        # Planar leader tracking
        leader_pos_cost = np.linalg.norm(leader_err[0:2]) ** 2
        leader_vel_cost = np.linalg.norm(leader_err[3:5]) ** 2

        # Planar formation costs
        f1_pos_cost = np.linalg.norm(follower1_err[0:2]) ** 2
        f2_pos_cost = np.linalg.norm(follower2_err[0:2]) ** 2

        f1_vel_cost = np.linalg.norm(follower1_err[3:5]) ** 2
        f2_vel_cost = np.linalg.norm(follower2_err[3:5]) ** 2

        # Small penalty on control effort
        control_cost = np.linalg.norm(U) ** 2

        if self.mode == "leader_only":
            total_cost = (
                self.w_leader_track * leader_pos_cost +
                self.w_vel * leader_vel_cost +
                self.w_control * control_cost
            )
        else:
            total_cost = (
                self.w_leader_track * leader_pos_cost +
                self.w_vel * leader_vel_cost +
                self.w_follower_form * (f1_pos_cost + f2_pos_cost) +
                self.w_vel * (f1_vel_cost + f2_vel_cost) +
                self.w_control * control_cost
            )
        return -0.01 * total_cost
        # return -total_cost

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.X = np.zeros(self.nx, dtype=np.float64)

        xr0 = self.leader_reference(0.0)

        def small_planar_noise():
            n = np.zeros(12, dtype=np.float64)

            # small planar perturbations
            n[0:2] = 0.05 * np.random.randn(2)   # x, y
            n[3:5] = 0.05 * np.random.randn(2)   # vx, vy

            # tiny vertical perturbations
            n[2] = 0.02 * np.random.randn()      # z
            n[5] = 0.02 * np.random.randn()      # vz

            # tiny angular perturbations
            n[6:12] = 0.01 * np.random.randn(6)

            return n

        # Leader near reference
        self.X[0:12] = xr0 + small_planar_noise()

        # Followers near desired formation
        self.X[12:24] = xr0 + self.d1 + small_planar_noise()
        self.X[24:36] = xr0 + self.d2 + small_planar_noise()

        obs = self.build_observation()
        info = {}

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_low, self.action_high)

        t = self.step_count * self.dt

        # Euler integration
        dX = self.dynamics(self.X, action, t)
        self.X = self.X + self.dt * dX

        self.step_count += 1

        reward = self.calculate_reward(action)
        obs = self.build_observation()

        terminated = False
        truncated = False

        # Blow-up protection
        if np.linalg.norm(self.X) > self.max_state_norm:
            terminated = True
            reward -= 1000.0

        if self.step_count >= self.episode_steps:
            truncated = True

        info = {}

        return obs, reward, terminated, truncated, info