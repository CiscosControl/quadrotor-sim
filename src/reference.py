import numpy as np

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
