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
        Q[0,0] = 100
        Q[1,1] = 100
        Q[2,2] = 100

    if R is None:
        R = 0.01*np.eye(m)

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
        Q = np.eye(n)

        # Stronger penalty on position states
        Q[0,0] = 500
        Q[1,1] = 500
        Q[2,2] = 500

    if R is None:
        R = 0.01*np.eye(m)

    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)

    # Compute gain
    K = np.linalg.inv(R) @ B.T @ P

    return K