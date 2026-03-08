import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Ensure these modules are in your local directory
from paramaters import QuadParams
from linear_model import linear_matrices
from controller import lqr_leader, lqr_follower

# -----------------------------
# 1. Reference Trajectory (Lissajous 2D)
# -----------------------------
def leader_reference(t):
    Xr = np.zeros(12)
    Xr_dot = np.zeros(12)
    
    # Parameters for a complex "Figure-Eight"
    A_x, A_y, omega = 10.0, 5.0, 0.2
    
    # Position (xr) - The "Spring" anchor point
    Xr[0] = A_x * np.sin(omega * t)      # x
    Xr[1] = A_y * np.sin(2 * omega * t)  # y
    Xr[2] = 2.0                          # Constant altitude z
    
    # Velocity (xr_dot) - Used for Feed-Forward (uff)
    # These derivatives tell the drone how fast it *should* be moving
    Xr_dot[0] = A_x * omega * np.cos(omega * t)
    Xr_dot[1] = 2 * A_y * omega * np.cos(2 * omega * t)
    
    # Heading: Facing direction of travel
    Xr[5] = np.arctan2(Xr_dot[1], Xr_dot[0]) 
    
    return Xr, Xr_dot
#-------------------------------------------
# 2. Define the Disturbance Function
#-------------------------------------------
def get_disturbance(t, state_dim=12):
    d = np.zeros(state_dim)
    # Scenario: A 2-second wind gust in the Y-direction at t=5.0
        # We apply it to the velocity/acceleration states (indices 6, 7, 8)
    d[7] = 1.5 # Acceleration disturbance in Y
    
    # Optional: Add continuous low-level noise
    noise = np.random.normal(0, 0.05, state_dim)
    return d + noise
# -----------------------------
# 2. Dynamics and Control (3 Drones)
# -----------------------------
def closed_loop_3drones(t, X, A, B, K_leader, K_f1, K_f2, B_pinv):
    # Split states for 3 drones (pi in the paper)
    x1, x2, x3 = X[0:12], X[12:24], X[24:36]

    # ---- Leader control (Feedback + Feed-Forward) ----
    xr1, xr1_dot = leader_reference(t)    
    
    # Feedback (ufb): Acts like a spring pulling to the reference
    u_fb = -K_leader @ (x1 - xr1)
    
    # Feed-forward (uff): Anticipates the curve to eliminate lag
    # Mathematical derivation: uff = B_pinv @ (xr_dot - A @ xr)
    u_ff = B_pinv @ (xr1_dot - A @ xr1)
    
    u1 = u_fb + u_ff

    # ---- Follower 1: 2m behind leader ----
    d1 = np.zeros(12)
    d1[0] = -2.0  # Stay 2m behind the leader's x-position
    e_rel1 = (x2 - x1) - d1 
    u2 = -K_f1 @ e_rel1

    # ---- Follower 2: 4m behind leader ----
    d2 = np.zeros(12)
    d2[0] = 4.0  # Stay 4m behind leader (2m behind F1)
    e_rel2 = (x3 - x1) - d2
    u3 = -K_f2 @ e_rel2
    # Calculate Disturbance
    dist1 = get_disturbance(t)
    dist2 = get_disturbance(t) # You can make these different for each drone!
    dist3 = get_disturbance(t)

    # Dynamics: x_dot = Ax + Bu + Disturbance
    x1_dot = A @ x1 + B @ u1 + dist1
    x2_dot = A @ x2 + B @ u2 + dist2
    x3_dot = A @ x3 + B @ u3 + dist3
    # Dynamics: x_dot = Ax + Bu

    return np.concatenate([x1_dot, x2_dot, x3_dot])

# -----------------------------
# 3. Setup and Simulation
# -----------------------------
param = QuadParams()
A, B = linear_matrices(param)
B_pinv = np.linalg.pinv(B) # Pre-calculate pseudoinverse for efficiency

K_leader = lqr_leader(A, B)
K_follower = lqr_follower(A, B)

# Initial conditions (Leader at origin, others already behind to avoid crossing)
X0 = np.zeros(36)
X0[0:2]   = [0.0, 0.0]    # Leader
X0[12:14] = [-3.0, 1.0]   # F1 (offset to avoid initial overlap)
X0[24:26] = [-6.0, -1.0]  # F2 (offset to avoid initial overlap)

t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

sol = solve_ivp(
    closed_loop_3drones, t_span, X0, t_eval=t_eval, 
    args=(A, B, K_leader, K_follower, K_follower, B_pinv)
)

# Extract positions for plotting
x1, y1 = sol.y[0,:], sol.y[1,:]
x2, y2 = sol.y[12,:], sol.y[13,:]
x3, y3 = sol.y[24,:], sol.y[25,:]

# Pre-calculate only the Xr portion of the reference for the "Ideal Path"
ref_path = np.array([leader_reference(t)[0] for t in t_eval])
ref_x, ref_y = ref_path[:, 0], ref_path[:, 1]

# -----------------------------
# 4. Analysis and Animation
# -----------------------------
# Calculate Euclidean distance to Leader
dist_L_F1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
dist_L_F2 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)

# Stability Plot
plt.figure(figsize=(10, 4))
plt.plot(t_eval, dist_L_F1, label='Leader to F1', color='green')
plt.plot(t_eval, dist_L_F2, label='Leader to F2', color='magenta')
plt.axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='Target 2m')
plt.axhline(y=4.0, color='m', linestyle='--', alpha=0.5, label='Target 4m')
plt.title("Formation Stability: Relative Distances (Feed-Forward Enabled)")
plt.xlabel("Time (s)"); plt.ylabel("Distance (m)"); plt.legend(); plt.grid(True)

# Animation
# Animation Setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-15, 15); ax.set_ylim(-10, 10)
ax.set_aspect('equal'); ax.grid(True)
ax.plot(ref_x, ref_y, 'r--', alpha=0.3, label="Reference Path")

# Trajectory Trails
line_l, = ax.plot([], [], 'b-', alpha=0.2)
line_f1, = ax.plot([], [], 'g-', alpha=0.2)
line_f2, = ax.plot([], [], 'm-', alpha=0.2)

# --- Virtual Springs ---
# These lines connect the drones to show the formation coupling
spring_l_f1, = ax.plot([], [], 'k-', linewidth=1, alpha=0.6, label="Spring L-F1")
spring_l_f2, = ax.plot([], [], 'k-', linewidth=1, alpha=0.4, label="Spring L-F2")

# Drone Markers
dot_l,  = ax.plot([], [], 'bo', markersize=8, label="Leader")
dot_f1, = ax.plot([], [], 'go', label="F1")
dot_f2, = ax.plot([], [], 'mo', label="F2")

ax.legend(loc='upper right')

def animate(i):
    # 1. Update Trajectory Trails
    line_l.set_data(x1[:i], y1[:i])
    line_f1.set_data(x2[:i], y2[:i])
    line_f2.set_data(x3[:i], y3[:i])
    
    # 2. Update Drone Positions
    dot_l.set_data([x1[i]], [y1[i]])
    dot_f1.set_data([x2[i]], [y2[i]])
    dot_f2.set_data([x3[i]], [y3[i]])
    
    # 3. Update Virtual Springs (Connecting Lines)
    # Line between Leader and Follower 1
    spring_l_f1.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    # Line between Leader and Follower 2
    spring_l_f2.set_data([x1[i], x3[i]], [y1[i], y3[i]])
    
    # Optional: Dynamic Spring Color (Turns red if stretched too far)
    # error = abs(dist_L_F1[i] - 2.0)
    # if error > 0.5: spring_l_f1.set_color('red')
    # else: spring_l_f1.set_color('black')

    return dot_l, dot_f1, dot_f2, line_l, line_f1, line_f2, spring_l_f1, spring_l_f2

ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)
plt.show()