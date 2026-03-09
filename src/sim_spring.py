import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Ensure these modules are in your local directory
from paramaters import QuadParams
from linear_model import linear_matrices
from controller import lqr_leader, lqr_follower
from reference import leader_reference

def get_repulsive_force(x_drone, y_drone, obstacles):
    """
    obstacles: list of [x, y, physical_radius, radius_of_influence]
    """
    f_rep = np.zeros(2)
    eta = 500.0  # Repulsion gain - tune this!
    
    for obs in obstacles:
        xo, yo, r_phys, rho0 = obs
        dist = np.sqrt((x_drone - xo)**2 + (y_drone - yo)**2)
        
        if dist < rho0:
            # Magnitude of repulsion
            mag = eta * (1.0/dist - 1.0/rho0) * (1.0/dist**2)
            # Direction (from obstacle to drone)
            unit_vec = np.array([x_drone - xo, y_drone - yo]) / dist
            f_rep += mag * unit_vec
            
    return f_rep

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

# Define obstacles: [x, y, radius]
#obstacles = [[5.0, 2.0, 0.5, 3.5], [-5.0, -2.0, 0.6,3.5]]
# Create 10 random obstacles within the plot bounds
obstacles = []
for _ in range(10):
    x = np.random.uniform(-5, 10)
    y = np.random.uniform(-5, 5)
    # Ensure they aren't exactly at the starting point [0, 0]
    if np.sqrt(x**2 + y**2) > 2.0:
        obstacles.append([x, y, 0.4, 2.5])
def closed_loop_3drones(t, X, A, B, K_leader, K_f1, K_f2, B_pinv):
    # Split states for 3 drones (pi in the paper)
    x1, x2, x3 = X[0:12], X[12:24], X[24:36]

    # --- 1. Calculate Repulsive Forces for each drone ---
    # We apply the force to the x and y acceleration channels
    # In a standard 12-state quadrotor, x/y acceleration are influenced by u
    f_rep1 = get_repulsive_force(x1[0], x1[1], obstacles)
    f_rep2 = get_repulsive_force(x2[0], x2[1], obstacles)
    f_rep3 = get_repulsive_force(x3[0], x3[1], obstacles)


    # ---- Leader control (Feedback + Feed-Forward) ----
    xr1, xr1_dot = leader_reference(t)    
    
    # Feedback (ufb): Acts like a spring pulling to the reference
    u_fb = -K_leader @ (x1 - xr1)
    
    # Feed-forward (uff): Anticipates the curve to eliminate lag
    # Mathematical derivation: uff = B_pinv @ (xr_dot - A @ xr)
    u_ff = B_pinv @ (xr1_dot - A @ xr1)
    
    u1 = u_fb + u_ff
    u1[0:2] += f_rep1

    # ---- Follower 1: 2m behind leader ----
    d1 = np.zeros(12)
    d1[0] = -2.0  # Stay 2m behind the leader's x-position
    e_rel1 = (x2 - x1) - d1 
    u2 = -K_f1 @ e_rel1
    u2[0:2] += f_rep2

    # ---- Follower 2: 4m behind leader ----
    d2 = np.zeros(12)
    d2[0] = 2.0  # Stay 4m behind leader (2m behind F1)
    e_rel2 = (x3 - x1) - d2
    u3 = -K_f2 @ e_rel2
    u3[0:2] += f_rep3

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
X0[24:26] = [-3.0, -1.0]  # F2 (offset to avoid initial overlap)

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


# --- INSERT THE OBSTACLE DRAWING LOOP HERE ---
for obs in obstacles:
    # obs[0]=x, obs[1]=y, obs[2]=phys_radius, obs[4]=radius_influence
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.2, label="Obstacle")
    ax.add_patch(circle)
# ---------------------------------------------



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

def plot_drone_states(sol, t_eval):
    # state_labels for a 12-state quadrotor
    labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)',
              'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 'p (rad/s)', 'q (rad/s)', 'r (rad/s)']
    
    drones = ['Leader', 'Follower 1', 'Follower 2']
    colors = ['b', 'g', 'm']
    
    # Create a figure with 4 rows (Pos, Att, Vel, Rates) and 3 columns (X, Y, Z / R, P, Y)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    fig.suptitle('Drone State Telemetry', fontsize=16)

    for i in range(12): # For each of the 12 states
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot each drone's corresponding state
        ax.plot(t_eval, sol.y[i, :], color=colors[0], label=drones[0] if i==0 else "")
        ax.plot(t_eval, sol.y[i + 12, :], color=colors[1], label=drones[1] if i==0 else "")
        ax.plot(t_eval, sol.y[i + 24, :], color=colors[2], label=drones[2] if i==0 else "")
        
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0: fig.legend(loc='upper right')
        if row == 3: ax.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Call this after your solve_ivp
plot_drone_states(sol, t_eval)



plt.figure(figsize=(10, 5))

# Follower 1 X-error (should stay at -2.0 relative to leader)
# Note: x1 is sol.y[0], x2 is sol.y[12]
plt.plot(t_eval, (sol.y[12, :] - sol.y[0, :]) - (-2.0), label='F1 X-Offset Error')
plt.plot(t_eval, (sol.y[24, :] - sol.y[0, :]) - (2.0), label='F2 X-Offset Error')

plt.title("Formation Keeping Error (Stationary Offset)")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.legend()
plt.grid(True)
plt.show()