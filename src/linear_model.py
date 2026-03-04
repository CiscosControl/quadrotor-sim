# linear_model.py

import numpy as np
def linear_matrices(params):
    m = params.m
    g = params.g

    Jx = params.J[0,0]
    Jy = params.J[1,1]
    Jz = params.J[2,2]

    A = np.zeros((12,12))
    B = np.zeros((12,4))

    # --- Position dynamics ---
    A[0,3] = 1   # x_dot = vx
    A[1,4] = 1   # y_dot = vy
    A[2,5] = 1   # z_dot = vz

    # --- Translational acceleration ---
    A[3,7] = -g  # vx_dot = -g * theta
    A[4,6] =  g  # vy_dot =  g * phi

    # --- Euler angle kinematics ---
    A[6,9]  = 1  # phi_dot   = wx
    A[7,10] = 1  # theta_dot = wy
    A[8,11] = 1  # psi_dot   = wz

    # --- Input matrix ---
    B[5,0]  = 1/m     # vz_dot = delta_T / m
    B[9,1]  = 1/Jx    # wx_dot = tau_x / Jx
    B[10,2] = 1/Jy    # wy_dot = tau_y / Jy
    B[11,3] = 1/Jz    # wz_dot = tau_z / Jz

    return A,B
