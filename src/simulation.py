# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from paramaters import QuadParams
from linear_model import linear_matrices
from multi_agent_model import system_model
from controller import lqr_coupled_gain


delta = 1.0
t_span = (0,10)
t_eval = np.linspace(0,10,1000)

def reference(t,delta):
    Xr = np.zeros(24)

    r = 5
    omega = 0.5

    xc = r * np.cos(omega*t)
    yc = r * np.sin(omega*t)

    Xr[0] = xc + delta/2
    Xr[1] = yc + delta/2

    Xr[12] = xc - delta/2
    Xr[13] = yc - delta/2

    return Xr 



def closed_loop_dynamics(t,X, M, Z, K):

    Xr = reference(t,delta)
    return (M - Z @ K) @ X +(Z@K)@Xr

param = QuadParams()
A, B = linear_matrices(param)
M, Z = system_model(A,B)


K = lqr_coupled_gain(M,Z,delta)

##Inintial conditionsc
X0 = np.zeros(24)

# Give drone 1 position offset
X0[0] = 15.0 #x1
X0[1] = 1.0 #y1

X0[12] = 15 #x2
X0[13] = 1.5 #y2


sol = solve_ivp(
    closed_loop_dynamics,
    t_span,
    X0,
    t_eval=t_eval,
    args=(M, Z, K)
)
# Solution of drone propagation
x1 = sol.y[0,:]
y1 = sol.y[1,:]

x2 = sol.y[12,:]
y2 = sol.y[13,:]

#
x1_ref = []
y1_ref = []

x2_ref = []
y2_ref = []

for t in t_eval:
    Xr =  reference(t,delta)
    x1_ref.append(Xr[0])
    y1_ref.append(Xr[1])
    x2_ref.append(Xr[12])
    y2_ref.append(Xr[13])



### Plots ###

#plot Drone trajectory
plt.plot(x1, y1, label="Drone 1")
plt.plot(x2, y2, label="Drone 2")

# plot reference positions as markers
plt.plot(x1_ref, y1_ref, 'r--', label = "Drone 1 Ref")
plt.plot(x2_ref, y2_ref, 'b--', label ="Drone 2 Ref")


plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("closed-loop trajectories")

plt.grid()
plt.show()


