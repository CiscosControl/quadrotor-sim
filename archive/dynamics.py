#dynamics.py

import numpy as np
import quaternion # here is function R
from paramaters import QuadParams

param = QuadParams() # inertia, mass and gravity of quadrotor

##states: position, velocity, quaternin, angular velocity

def dynamics(t,x, u, w, param):

    #states
    pos = x[0:3]
    v = x[3:6]
    phi, theta, psi = x[6:9]
    omega = x[9:12]

    # inputs
    T = u[0]
    torque = u[1:4]    

    # Rotation from body to inertial frame
    R = quaternion.rotation_matrix(phi,theta, psi)

    #forces: thrust, weight, drag
    f_T_body = np.array([0,0,T])
    f_T = R @ f_T_body
    f_g = ([0, 0, -param.g*param.m])

    f_net = f_T + f_g


    # Translational dynamics
    pos_dot = v
    v_dot = f_net /param.m

    #Euler kinamatics

    angle_dot = quaternion.euler_rate_matrix(phi,theta)@omega
    

    # Rotational Dynamics
    omega_dot = np.linalg.inv(param.J)@(torque - np.cross(omega,(param.J@omega)))



    x_dot = [pos_dot, v_dot, angle_dot, omega_dot]

    return x_dot

