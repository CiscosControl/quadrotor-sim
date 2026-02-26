from parameters import QuadParams
from linear_model import linear_matrices
import numpy as np
from scipy.linalg import solve_continuous_are
def lqr(Q,R):

    """
    Solve th continous time lqr controller
    dx/dt = A x + B
    """

    param = QuadParams()
    A,B = linear_matrices(param)

    P = solve_continuous_are(A,B,Q,R)

    K = np.linalg.inv(R) @ B.T @ P


    return K