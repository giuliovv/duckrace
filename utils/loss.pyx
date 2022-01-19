# cython: infer_types=True

import numpy as np

from utils.utils import my_odometry

def mpc_loss(u, int N, float x0, float y0, float theta0, float v0, float w0, int closest_index, Q, R, float dt, last_actions, traj):
    """
    L(u) = sum_{t=0}^{N-1} Q * ||x_o_t - x_t||^2 + R * ||u_t||^2
    """
    u = u.reshape(-1, 2)
    u = np.concatenate((last_actions, u))
    cdef loss = 0
    for t in range(N-1):
        p, v0, w0 = my_odometry(u[t], x0, y0, theta0, v0, w0, dt=dt)
        loss += Q[t] * ((p.x - traj[closest_index,0])**2 + (p.y - traj[closest_index,1])**2) + R * (u[t][0]**2 + u[t][1]**2)
        x0, y0, theta0 = p.x, p.y, p.theta
    return loss
