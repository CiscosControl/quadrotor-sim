# quaternion.py
import jax.numpy as jnp

def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion conjugate for q = [w, x, y, z]."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product p ⊗ q for quaternions in [w, x, y, z] form."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate 3-vector v from body to inertial using quaternion q (body->inertial).
    v is a 3-vector. Returns a 3-vector.
    """
    v_quat = jnp.concatenate([jnp.array([0.0]), v])
    v_rot = quat_mul(quat_mul(q, v_quat), quat_conj(q))
    return v_rot[1:]

def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize quaternion to unit length (safe for small norms)."""
    n = jnp.linalg.norm(q)
    n = jnp.maximum(n, eps)
    return q / n


def rotation_matrix(phi, theta, psi):
    """
    ZYX Rotation Matrix (Body -> Inertial)
    """

    ctheta = np.cos(theta)
    cphi   = np.cos(phi)
    cpsi   = np.cos(psi)

    stheta = np.sin(theta)
    sphi   = np.sin(phi)
    spsi   = np.sin(psi)

    R = np.array([
        [ctheta*cpsi,
         ctheta*spsi,
         -stheta],

        [sphi*stheta*cpsi - cphi*spsi,
         sphi*stheta*spsi + cphi*cpsi,
         sphi*ctheta],

        [cphi*stheta*cpsi + sphi*spsi,
         cphi*stheta*spsi - sphi*cpsi,
         cphi*ctheta]
    ])

    return R

def euler_rate_matrix(phi, theta):
    """
    Returns T(phi, theta) matrix such that:
    euler_dot = T @ omega
    """

    sphi = np.sin(phi)
    cphi = np.cos(phi)
    ttheta = np.tan(theta)
    ctheta = np.cos(theta)

    T = np.array([
        [1, sphi*ttheta, cphi*ttheta],
        [0, cphi, -sphi],
        [0, sphi/ctheta, cphi/ctheta]
    ])

    return T