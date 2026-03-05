import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from paramaters import QuadParams
from linear_model import linear_matrices
from controller import lqr_single


delta = 1.0
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)


def leader_reference(t):
    r = 5
    omega = 0.5

    Xr = np.zeros(12)
    Xr[0] = r * np.cos(omega * t)
    Xr[1] = r * np.sin(omega * t)

    return Xr


def closed_loop(t, X, A, B, K_leader, K_follower):

    # Split states
    x1 = X[0:12]      # leader
    x2 = X[12:24]     # follower

    # ----- Leader control -----
    xr1 = leader_reference(t)
    u1 = -K_leader @ (x1 - xr1)

    # ----- Follower control -----
    # Desired relative offset
    desired_offset = np.zeros(12)
    desired_offset[0] = -delta  # follower stays delta behind in x

    xr2 = x1 + desired_offset
    u2 = -K_follower @ (x2 - xr2)

    # Dynamics
    x1_dot = A @ x1 + B @ u1
    x2_dot = A @ x2 + B @ u2

    return np.concatenate([x1_dot, x2_dot])


# Build system
param = QuadParams()
A, B = linear_matrices(param)

# LQR gains (same A,B but separate controllers)
K_leader = lqr_single(A, B)
K_follower = lqr_single(A, B)

# Initial conditions
X0 = np.zeros(24)

X0[0] = 15.0
X0[1] = 1.0

X0[12] = 10.0
X0[13] = 1.5


sol = solve_ivp(
    closed_loop,
    t_span,
    X0,
    t_eval=t_eval,
    args=(A, B, K_leader, K_follower)
)

# Extract positions
x1 = sol.y[0,:]
y1 = sol.y[1,:]

x2 = sol.y[12,:]
y2 = sol.y[13,:]

# Plot
plt.plot(x1, y1, label="Leader")
plt.plot(x2, y2, label="Follower")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Leader–Follower LQR")
plt.grid()
plt.show()