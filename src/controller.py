# Controller.py

import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_leader(A, B, Q=None, R=None):
    """
    Standard infinite-horizon LQR for a single drone
    """

    n = A.shape[0]
    m = B.shape[1]

    if Q is None:
        Q = np.eye(n)

        # Stronger penalty on position states
        Q[0,0] = 50
        Q[1,1] = 50
        Q[2,2] = 50

    if R is None:
        R = np.eye(m)

    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)

    # Compute gain
    K = np.linalg.inv(R) @ B.T @ P

    return K

def lqr_follower(A, B, Q=None, R=None):
    """
    Standard infinite-horizon LQR for a single drone
    """

    n = A.shape[0]
    m = B.shape[1]

    if Q is None:
        Q = np.eye(n)* 10.0

        # Stronger penalty on position states
        Q[0,0] = Q[1,1] = Q[2,2] = 40.0  # High position penalty
        Q[3,3] = Q[4,4] = Q[5,5] = 60.0
        Q[6,6] = Q[7,7] = Q[8,8] = 30.0   # ADD THIS: Velocity damping!
        Q[9,9] = 100.0 
    if R is None:
        R = np.eye(m)*10.0

    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)

    # Compute gain
    K = np.linalg.inv(R) @ B.T @ P

    return K