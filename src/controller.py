# controller.py
import numpy as np
from scipy.linalg import solve_continuous_are


def _build_rel_selectors_12():
    """
    For a single 12-state drone:
      x = [x,y,z,vx,vy,vz,phi,theta,psi,wx,wy,wz]
    Build selectors for position p and velocity v.
    """
    C_p = np.zeros((3, 12))
    C_p[0, 0] = 1.0
    C_p[1, 1] = 1.0
    C_p[2, 2] = 1.0

    C_v = np.zeros((3, 12))
    C_v[0, 3] = 1.0
    C_v[1, 4] = 1.0
    C_v[2, 5] = 1.0

    return C_p, C_v


def _build_coupled_Q_2drones(Qp: np.ndarray, Qv: np.ndarray) -> np.ndarray:
    """
    Build Q (24x24) that penalizes relative position and relative velocity:
      e_p = p1 - p2
      e_v = v1 - v2
    Cost: e_p^T Qp e_p + e_v^T Qv e_v

    This is the Option B2 coupling mechanism: Q introduces cross terms.
    """
    Qp = np.asarray(Qp); Qv = np.asarray(Qv)
    if Qp.shape != (3, 3) or Qv.shape != (3, 3):
        raise ValueError("Qp and Qv must be 3x3.")

    C_p, C_v = _build_rel_selectors_12()

    # Relative maps on stacked X=[x1;x2] in R^24
    C_rel_p = np.block([C_p, -C_p])  # 3x24
    C_rel_v = np.block([C_v, -C_v])  # 3x24

    Q = C_rel_p.T @ Qp @ C_rel_p + C_rel_v.T @ Qv @ C_rel_v
    return Q


def lqr_coupled_gain(M: np.ndarray,
                     Z: np.ndarray,
                     delta: float,
                     Qp_weight: float = 50.0,
                     Qv_weight: float = 10.0,
                     R_diag: np.ndarray | None = None) -> np.ndarray:
    """
    Computes the Option B2 LQR gain K for the stacked 2-drone system:

      Xdot = M X + Z U
      X in R^24, U in R^8

    The coupling is introduced through Q, which penalizes relative position/velocity:
      e_p = p1 - p2 - [delta, 0, 0]^T   (offset handled via reference shift in simulation)
      e_v = v1 - v2

    IMPORTANT:
    - LQR gain K itself does NOT depend on the constant offset delta; delta only affects X_ref.
    - We accept delta here only to keep your API aligned with params.py / your project.
      (You will use delta later to build X_ref when applying u = -K (X - X_ref).)

    Returns:
      K (8x24) such that U = -K X for the regulation problem.
    """

    M = np.asarray(M); Z = np.asarray(Z)
    n = M.shape[0]
    m = Z.shape[1]

    if M.shape != (n, n):
        raise ValueError("M must be square.")
    if Z.shape[0] != n:
        raise ValueError("Z must have same number of rows as M.")
    if n != 24 or m != 8:
        raise ValueError(f"Expected (M,Z) dimensions (24x24, 24x8), got {M.shape}, {Z.shape}")

    # Build coupled Q on relative position/velocity (3D)
    Qp = Qp_weight * np.eye(3)
    Qv = Qv_weight * np.eye(3)
    Q = _build_coupled_Q_2drones(Qp, Qv)  # 24x24

    # Input cost
    if R_diag is None:
        R = np.diag([1.0] * m)  # default
    else:
        R_diag = np.asarray(R_diag).reshape(m,)
        R = np.diag(R_diag)

    # Solve CARE and compute K
    P = solve_continuous_are(M, Z, Q, R)
    K = np.linalg.inv(R) @ (Z.T @ P)  # 8x24

    return K