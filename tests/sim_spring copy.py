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
    f_rep = np.zeros(2)
    eta = 15.0  # Still strong, but more reasonable
    max_force = 20.0 # Absolute limit to prevent numerical explosions
    
    for obs in obstacles:
        xo, yo, r_phys, rho0 = obs
        dist = np.sqrt((x_drone - xo)**2 + (y_drone - yo)**2)
        
        # Prevent division by zero mathematically
        dist = max(dist, 0.01) 
        
        if dist < rho0:
            mag = eta * (1.0/dist - 1.0/rho0) * (1.0/dist**2)
            # SATURATE THE FORCE
            mag = min(mag, max_force) 
            
            unit_vec = np.array([x_drone - xo, y_drone - yo]) / dist
            f_rep += mag * unit_vec
            
    return f_rep

#-------------------------------------------
# 2. Define the Disturbance Function
#-------------------------------------------
def get_disturbance(t, state_dim=12):
    d = np.zeros(state_dim)
    # Wind gust scenario
    if 4.9 < t < 5.1:
        d[7] = 1.5 
    
    # Only add noise to linear velocities (indices 6, 7, 8) 
    # and angular rates (indices 9, 10, 11)
    noise = np.zeros(state_dim)
    noise[6:12] = np.random.normal(0, 0.01, 6)
    
    return d + noise
# -----------------------------
# 2. Dynamics and Control (3 Drones)
# -----------------------------

# Define obstacles: [x, y, radius]
#obstacles = [[5.0, 2.0, 0.5, 3.5], [-5.0, -2.0, 0.6,3.5]]
# Create 10 random obstacles within the plot bounds
obstacles = []
for _ in range(15):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-5, 5)
    # Ensure they aren't exactly at the starting point [0, 0]
    if np.sqrt(x**2 + y**2) > 4.0:
        obstacles.append([x, y, 0.3, 1.5])
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
    e_rel = x1 - xr1

    e_rel[0:2] -= f_rep1
    u_fb = -K_leader @ e_rel
    
    # Feed-forward (uff): Anticipates the curve to eliminate lag
    # Mathematical derivation: uff = B_pinv @ (xr_dot - A @ xr)
    u_ff = B_pinv @ (xr1_dot - A @ xr1)
    
    u1 = u_fb + u_ff
    ##u1[0:2] += f_rep1

    # ---- Follower 1: 2m behind leader ----
    d1 = np.zeros(12)
    d1[0] = -2.0  # Stay 2m behind the leader's x-position
    e_rel1 = (x2 - x1) - d1 
    e_rel1[0:2] -= f_rep2
    u2 = -K_f1 @ e_rel1
    #u2[0:2] += f_rep2

    # ---- Follower 2: 4m behind leader ----
    d2 = np.zeros(12)
    d2[0] = 2.0  # Stay 4m behind leader (2m behind F1)
    e_rel2 = (x3 - x1) - d2
    e_rel2[0:2] -= f_rep3
    u3 = -K_f2 @ e_rel2
    #u3[0:2] += f_rep3

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
X0[12:14] = [-2.0, 0.0]   # F1 (offset to avoid initial overlap)
X0[24:26] = [ 2.0, 0.0]  # F2 (offset to avoid initial overlap)
time = 90
t_span = (0,time)
t_eval = np.linspace(0, time, 1000)

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
ax.set_xlim(-15, 15); ax.set_ylim(-7.5, 7.5)
ax.set_aspect('equal'); ax.grid(True)


# --- UPDATED OBSTACLE DRAWING LOOP ---
for idx, obs in enumerate(obstacles):
    # obs[0]=x, obs[1]=y, obs[2]=phys_radius, obs[3]=rho0 (influence radius)
    xo, yo, r_phys, rho0 = obs
    
    # 1. Plot the Physical Obstacle (Solid)
    label_phys = 'Obstacle' if idx == 0 else '_nolegend_'
    circle_phys = plt.Circle((xo, yo), r_phys, color='r', alpha=0.5, label=label_phys)
    ax.add_patch(circle_phys)
    
    # 2. Plot the Radius of Influence (rho0)
    label_rho = 'Influence (rho0)' if idx == 0 else '_nolegend_'
    circle_rho = plt.Circle((xo, yo), rho0, color='r', fill=False, 
                            linestyle='--', alpha=0.3, label=label_rho)
    ax.add_patch(circle_rho)
# -------------------------------------



ax.plot(ref_x, ref_y, 'r--', alpha=0.3, label="Reference Path")

# Trajectory Trails
line_l, = ax.plot([], [], 'b-', alpha=0.2)
line_f1, = ax.plot([], [], 'g-', alpha=0.2)
line_f2, = ax.plot([], [], 'm-', alpha=0.2)

# --- Virtual Springs ---
# These lines connect the drones to show the formation coupling
spring_l_f1, = ax.plot([], [], 'k-', linewidth=1, alpha=0.6, label="Link")
spring_l_f2, = ax.plot([], [], 'k-', linewidth=1, alpha=0.6)

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

def plot_drone_states(sol, t_eval):
    # state_labels for a 12-state quadrotor
    # Convert angular states (angles and rates) to degrees/deg/s for plotting
    labels = ['X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',
              'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)', 'omega_x (deg/s)', 'omega_y (deg/s)', 'omega_z (deg/s)']
    drones = ['Leader', 'Follower 1', 'Follower 2']
    colors = ['b', 'g', 'm']
    
    # Create a figure with 4 rows (Pos, Att, Vel, Rates) and 3 columns (X, Y, Z / R, P, Y)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    fig.suptitle('Drone State Telemetry', fontsize=16)

    for i in range(12): # For each of the 12 states
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Determine if this state is an angle or angular rate that needs conversion
        if i in (6, 7, 8, 9, 10, 11):
            # indices 3-5: Roll, Pitch, Yaw (rad -> deg)
            # indices 9-11: p, q, r (rad/s -> deg/s)
            s0 = np.rad2deg(sol.y[i, :])
            s1 = np.rad2deg(sol.y[i + 12, :])
            s2 = np.rad2deg(sol.y[i + 24, :])
        else:
            s0 = sol.y[i, :]
            s1 = sol.y[i + 12, :]
            s2 = sol.y[i + 24, :]

        # Plot each drone's corresponding state
        ax.plot(t_eval, s0, color=colors[0], label=drones[0] if i==0 else "")
        ax.plot(t_eval, s1, color=colors[1], label=drones[1] if i==0 else "")
        ax.plot(t_eval, s2, color=colors[2], label=drones[2] if i==0 else "")

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

fps = 50
ani.save('drone_formation.gif', writer='pillow', fps=fps)

print("Save complete!")
plt.show()