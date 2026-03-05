import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_single(A, B, Q=None, R=None):
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