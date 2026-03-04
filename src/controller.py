import numpy as np
from scipy.linalg import solve_continuous_are


def _build_rel_selectors_12():
    C_p = np.zeros((3, 12))
    C_p[0, 0] = 1.0
    C_p[1, 1] = 1.0
    C_p[2, 2] = 1.0

    C_v = np.zeros((3, 12))
    C_v[0, 3] = 1.0
    C_v[1, 4] = 1.0
    C_v[2, 5] = 1.0

    return C_p, C_v


def _build_coupled_Q_2drones(Qp, Qv):
    C_p, C_v = _build_rel_selectors_12()

    C_rel_p = np.block([C_p, -C_p])  # 3x24
    C_rel_v = np.block([C_v, -C_v])  # 3x24

    Q = C_rel_p.T @ Qp @ C_rel_p + C_rel_v.T @ Qv @ C_rel_v
    return Q


def lqr_coupled_gain(M,
                     Z,
                     delta,
                     Qp_weight=80.0,
                     Qv_weight=20.0,
                     Qtrack_weight=40.0,
                     R_diag=None):

    M = np.asarray(M)
    Z = np.asarray(Z)

    n = M.shape[0]
    m = Z.shape[1]

    # ---------- Formation coupling ----------
    Qp = Qp_weight * np.eye(3)
    Qv = Qv_weight * np.eye(3)
    Q_form = _build_coupled_Q_2drones(Qp, Qv)

    # ---------- Absolute tracking penalty ----------
    Q_track = np.zeros((24, 24))

    # Penalize position tracking for drone 1
    Q_track[0, 0] = Qtrack_weight
    Q_track[1, 1] = Qtrack_weight
    Q_track[2, 2] = Qtrack_weight

    # Penalize position tracking for drone 2
    Q_track[12, 12] = Qtrack_weight
    Q_track[13, 13] = Qtrack_weight
    Q_track[14, 14] = Qtrack_weight

    # Total Q
    Q = Q_form + Q_track

    # ---------- Input cost ----------
    if R_diag is None:
        R = np.eye(m)
    else:
        R = np.diag(np.asarray(R_diag).reshape(m,))

    # ---------- Solve CARE ----------
    P = solve_continuous_are(M, Z, Q, R)

    K = np.linalg.inv(R) @ Z.T @ P

    return K