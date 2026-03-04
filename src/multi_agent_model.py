# multi_agent_model.py

from linear_model import linear_matrices
from paramaters import QuadParams
import numpy as np

def system_model(A,B):
    """
    Model the coupled UAVs
    dX/dt = MX + Zu
    """

    M = np.block([
        [A, np.zeros_like(A)],
        [np.zeros_like(A),A]
    ])

    Z = np.block([
        [B, np.zeros_like(B)],
        [np.zeros_like(B),B]
    ])


    return M,Z
