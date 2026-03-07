import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Ensure these modules are in your local directory
from paramaters import QuadParams
from linear_model import linear_matrices
from controller import lqr_leader
from controller import lqr_follower

# -----------------------------
# 1. Reference Trajectory (Lissajous 2D)
# -----------------------------
def leader_reference(t):
    Xr = np.zeros(12)
    A_x, A_y, omega = 10.0, 5.0, 0.2
    
    Xr[0] = A_x * np.sin(omega * t)      # x
    Xr[1] = A_y * np.sin(2 * omega * t)  # y
    Xr[2] = 2.0                          # z
    
    # Velocity feed-forward (p_dot)
    Xr[6] = A_x * omega * np.cos(omega * t)
    Xr[7] = 2 * A_y * omega * np.cos(2 * omega * t)
    
    Xr[5] = np.arctan2(Xr[7], Xr[6])     # yaw
    return Xr

# -----------------------------
# 2. Dynamics and Control
# -----------------------------
def closed_loop_3drones(t, X, A, B, K_leader, K_f1, K_f2):
    x1, x2, x3 = X[0:12], X[12:24], X[24:36]

    # Leader tracks global reference
    xr1 = leader_reference(t)
    #u1 = -K_leader @ (x1 - xr1)
    # numerical derivative of reference
    dt = 1e-3
    xr1_dot = (leader_reference(t+dt) - xr1)/dt

    u_ff = np.linalg.pinv(B) @ (xr1_dot - A @ xr1)

    u1 = -K_leader @ (x1 - xr1) + u_ff

    # Follower 1: 2m behind leader
    d1 = np.zeros(12); d1[0] = 2.0
    u2 = -K_f1 @ ((x2 - x1) - d1)

    # Follower 2: 4m behind leader
    d2 = np.zeros(12); d2[0] = -2.0
    u3 = -K_f2 @ ((x3 - x1) - d2)

    return np.concatenate([A@x1 + B@u1, A@x2 + B@u2, A@x3 + B@u3])

# -----------------------------
# 3. Setup
# -----------------------------
param = QuadParams()
A, B = linear_matrices(param)
K_leader = lqr_leader(A, B)
K_follower = lqr_follower(A,B)


# Collision-free Initial positions
X0 = np.zeros(36)
X0[0:2] = [0, 0]; X0[12:14] = [-3, 1]; X0[24:26] = [-6, -1]

t_span = (0, 20); t_eval = np.linspace(0, 20, 1000)
sol = solve_ivp(closed_loop_3drones, t_span, X0, t_eval=t_eval, args=(A, B, K_leader, K_follower, K_follower))

x1, y1 = sol.y[0,:], sol.y[1,:]
x2, y2 = sol.y[12,:], sol.y[13,:]
x3, y3 = sol.y[24,:], sol.y[25,:]

# Pre-calculate the full reference trajectory for the plot
ref_points = np.array([leader_reference(t) for t in t_eval])
ref_x, ref_y = ref_points[:, 0], ref_points[:, 1]

# -----------------------------
# 4. Data Extraction & Plotting |Animation with Reference
# -----------------------------

# Inter-drone distance calculation
dist_L_F1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
dist_L_F2 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)

# Distance Plot
plt.figure(figsize=(10, 4))
plt.plot(t_eval, dist_L_F1, label='Leader to F1', color='green')
plt.plot(t_eval, dist_L_F2, label='Leader to F2', color='magenta')
plt.axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='Target 2m')
plt.axhline(y=4.0, color='m', linestyle='--', alpha=0.5, label='Target 4m')
plt.title("Formation Stability: Distances to Leader")
plt.xlabel("Time (s)"); plt.ylabel("Distance (m)"); plt.legend(); plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-15, 15); ax.set_ylim(-10, 10)
ax.set_aspect('equal'); ax.grid(True)

# Static Reference Trajectory (The target path)
ax.plot(ref_x, ref_y, 'r--', alpha=0.3, label="Ideal Path")

line_l, = ax.plot([], [], 'b-', alpha=0.3)
line_f1, = ax.plot([], [], 'g-', alpha=0.3)
line_f2, = ax.plot([], [], 'm-', alpha=0.3)
dot_l, = ax.plot([], [], 'bo', label="Leader")
dot_f1, = ax.plot([], [], 'go', label="F1")
dot_f2, = ax.plot([], [], 'mo', label="F2")
ax.legend(loc='upper right')

def animate(i):
    # Current trail lines
    line_l.set_data(x1[:i], y1[:i])
    line_f1.set_data(x2[:i], y2[:i])
    line_f2.set_data(x3[:i], y3[:i])
    # Current drone positions
    dot_l.set_data([x1[i]], [y1[i]])
    dot_f1.set_data([x2[i]], [y2[i]])
    dot_f2.set_data([x3[i]], [y3[i]])
    return dot_l, dot_f1, dot_f2, line_l, line_f1, line_f2

ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)
plt.show()