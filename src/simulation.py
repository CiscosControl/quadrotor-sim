import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from paramaters import QuadParams
from linear_model import linear_matrices
from multi_agent_model import system_model
from controller import lqr_coupled_gain


def closed_loop_dynamics(t,X, M, Z, K):
    return (M - Z @ K) @ X

param = QuadParams()
A, B = linear_matrices(param)
M, Z = system_model(A,B)


delta = 2.0

K = lqr_coupled_gain(M,Z,delta)

##Inintial conditionsc
X0 = np.zeros(24)

#give drone 1 position offset
X0[0] = 2.0 #x1
X0[1] = 1.0 #y1

X0[12] = -1.5 #x2
X0[13] = 1.5 #y2

t_span = (0,10)
t_eval = np.linspace(0,10,1000)

sol = solve_ivp(
    closed_loop_dynamics,
    t_span,
    X0,
    t_eval=t_eval,
    args=(M, Z, K)
)

x1 = sol.y[0,:]
y1 = sol.y[1,:]

x2 = sol.y[12,:]
y2 = sol.y[13,:]

plt.plot(x1, y1, label = "Drone 1")
plt.plot(x2, y2, label = "Drone 2")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("closed-loop trajectories")

plt.grid()
plt.show()


