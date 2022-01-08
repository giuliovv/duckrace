from typing import Tuple

import control as ct
import cv2
import geometry
import numpy as np
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass


@dataclass
class Position:
    x: float = 0
    y: float = 0
    theta: float = 0

    def __add__(self, other):
        x1, y1, theta1 = self.x, self.y, self.theta
        x2, y2, theta2 = other.x, other.y, other.theta
        return Position(x1+x2, y1+y2, (theta1+theta2)%360)

    def from_friendly(self, p_string: str):
        """
        Read geometry.SE2.friendly
        """
        self.theta = float(p_string.split("Pose(Rot(")[1].split("deg)")[0])
        self.x = float(p_string.split("[")[1].replace("  ", " ").split(" ")[0])
        self.y = float(p_string.split("[")[1].replace("  ", " ").split(" ")[1].replace("])", ""))
        return

@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float

@dataclass
class PWMCommands:
    """
    PWM commands are floats between -1 and 1.
    """
    motor_left: float
    motor_right: float

def get_acceleration(action, u=None, w=None):
    """
    Second derivative of x

    :param action: the action to be taken [wl, wr]
    :param u: the current velocity
    :param w: the current angular velocity

    :return: (acc longitudinal, acc angular)
    """
    # https://drive.google.com/file/d/19U1DUo3GtqHxncEKLn2d6RRdLTLgD0Bv/view
    # Paragraph 5.1.5

    u1 = 5
    u2 = 0
    u3 = 0
    w1 = 4
    w2 = 0
    w3 = 0
    # parameters for forced dynamics
    uar = 1.5
    ual = 1.5
    war = 15  # modify this for trim
    wal = 15

    u_alpha_r = uar
    u_alpha_l = ual
    w_alpha_r = war
    w_alpha_l = wal
    
    # wr, wl
    U = np.array([action[1], action[0]])
    V = U.reshape(U.size, 1)
    V = np.clip(V, -1, +1)

    # Previous step linear and angular speed
    # u, w = longit_prev, angular_prev

    ## Calculate Dynamics
    # nonlinear Dynamics - autonomous response
    f_dynamic = np.array([[-u1 * u - u2 * w + u3 * w ** 2], [-w1 * w - w2 * u - w3 * u * w]])
    # input Matrix
    B = np.array([[u_alpha_r, u_alpha_l], [w_alpha_r, -w_alpha_l]])
    # forced response
    f_forced = np.matmul(B, V)
    # acceleration
    x_dot_dot = f_dynamic + f_forced
    return x_dot_dot

def get_dimensions(env):
    """
    Get the dimensions of the environment.
    """
    return (env.grid_height*env.road_tile_size, env.grid_width*env.road_tile_size)

def get_position(env) -> Position:
    """
    Get the position of the agent in the environment.
    """
    p_se_matrix = env.cartesian_from_weird(env.cur_pos, env.cur_angle)
    p_string = geometry.SE2.friendly(p_se_matrix)
    p = Position()
    p.from_friendly(p_string)
    return p

def get_trajectory(env, no_preprocessing=False, samples=50, scaled=True):
    """
    Get the trajectory of the agent in the environment.

    :param no_preprocessing: if True, the trajectory is not preprocessed
    :param samples: the number of samples to take
    :param scaled: if True, the trajectory is scaled to the environment size

    :return: np.array
    """
    env.reset()
    top_view = np.flip(env.render(mode="top_down"), [0])[35:-30,130:-130]

    img_hsv = cv2.cvtColor(top_view, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(top_view, cv2.COLOR_RGB2GRAY)

    lower_yellow = np.array([20,100,150])
    upper_yellow = np.array([30,255,255])

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_and(gray, mask_yellow)

    if not no_preprocessing:
        kernel = np.ones((4, 4), np.uint8)
        eroded = cv2.erode(mask, kernel) 

        low_threshold = 89
        high_threshold = 80
        edges = cv2.Canny(eroded, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 3  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 50  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        _points = lines.reshape(-1, 2)

        x, y = _points.T

    else:
        x_all, y_all = np.nonzero(mask)
        x, y = x_all, y_all

    x_sorted, y_sorted = sort_xy(x, y)

    points = np.array([x_sorted, y_sorted]).T

    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Build a list of the spline function, one for each dimension:
    # k is the degree of the spline. 3 means cubic

    splines = [UnivariateSpline(distance, coords, k=1, s=.01) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, samples)
    points_fitted = np.vstack( spl(alpha) for spl in splines ).T

    if scaled:
        points_fitted[:,0] = points_fitted[:,0]*env.grid_width*env.road_tile_size/top_view.shape[1]
        points_fitted[:,1] = points_fitted[:,1]*env.grid_height*env.road_tile_size/top_view.shape[0]
        return points_fitted

    return points_fitted
        

def my_odometry(action, x0, y0, theta0, v0=0, w0=0, dt=0.033)-> Tuple[Position, float, float]:
    """
    Calculate the odometry from the action and the current state.

    :param action: the action to perform
    :param x0: the initial x position
    :param y0: the initial y position
    :param theta0: the initial orientation
    :param v0: the initial linear speed
    :param w0: the initial angular speed
    :param dt: the time step

    :return: (Position, float, float)
    """
    x_dot_dot, w_dot_dot = get_acceleration(action, u=v0, w=w0)

    v1 = v0 + x_dot_dot[0]*dt
    w1 = w0 + w_dot_dot[0]*dt

    # Runge Kutta
    x1 = x0 + v0*dt*np.cos(theta0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(theta0 + w0*dt/2)
    theta1 = theta0 + w0*dt

    return Position(x1, y1, theta1), v1, w1

def my_odometry_linearized(action, x0, y0, theta0, v0=0, w0=0, dt=0.033)-> Tuple[Position, float, float]:
    """
    my_odometry but linearized.

    :param action: the action to perform
    :param x0: the initial x position
    :param y0: the initial y position
    :param theta0: the initial orientation
    :param v0: the initial linear speed
    :param w0: the initial angular speed
    :param dt: the time step

    :return: (Position, float, float)
    """
    x_dot_dot, w_dot_dot = get_acceleration(action, u=v0, w=w0)

    v1 = v0 + x_dot_dot[0]*dt
    w1 = w0 + w_dot_dot[0]*dt

    # Runge Kutta
    x1 = x0 + v0*dt
    y1 = y0 + v0*dt*(theta0 + w0*dt/2 )
    theta1 = theta0 + w0*dt

    return Position(x1, y1, theta1), v1, w1

def my_odom_lin(v, w, x0, y0, t0, dt, v_eq=0, w_eq=0):
    u = np.array([v, w]).T

    B_00 = dt*np.cos(t0+w_eq*dt/2)
    B_01 = -v_eq*dt*np.sin(t0+w_eq*dt/2)*dt/2
    B_10 = dt*np.sin(t0+w_eq*dt/2)
    B_11 = v_eq*dt*np.cos(t0+w_eq*dt/2)*dt/2
    B_20 = 0
    B_21 = w_eq*dt

    B = np.array([[B_00, B_01], [B_10, B_11], [B_20, B_21]])

    A0 = np.array([x0, y0, -t0]).T
    A1 = -B@np.array([v_eq, w_eq]).T

    A = A0 + A1

    x_k = A+B@u

    return x_k

def sort_xy(x, y):
    """
    Sort by angle
    """

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return x_sorted, y_sorted

#################
# UNDERGOING TEST

# def auto_odom_full(action, x0, y0, theta0, v0=0, w0=0, dt=0.033):
#     # Trick is to +1 delay
#     x_dot_dot, w_dot_dot = get_acceleration(action, u=v0, w=w0)
#     v1 = v0 + x_dot_dot[0]*dt
#     w1 = w0 + w_dot_dot[0]*dt

#     # Runge Kutta
#     x1 = x0 + v1*dt*np.cos(theta0 + w1*dt/2)
#     y1 = y0 + v1*dt*np.sin(theta0 + w1*dt/2)
#     theta1 = theta0 + w1*dt

#     return Position(x1, y1, theta1), v1, w1

def auto_odom_full(t, x, u, params):
    # Trick is to +1 delay
    dt = params.get('dt', 0.033)

    wl, wr = u[0], u[1]

    x0, y0, theta0, v0, w0 = x[0], x[1], x[2], x[3], x[4]

    x_dot_dot, w_dot_dot = get_acceleration([wl, wr], u=v0, w=w0)
    v1 = v0 + x_dot_dot[0]*dt
    w1 = w0 + w_dot_dot[0]*dt

    # Runge Kutta
    x1 = x0 + v1*dt*np.cos(theta0 + w1*dt/2)
    y1 = y0 + v1*dt*np.sin(theta0 + w1*dt/2)
    theta1 = theta0 + w1*dt

    return [x1, y1, theta1, v1, w1]

def linearized_odom(action, x0, y0, theta0, v0=0, w0=0, dt=0.033, return_result=False):
    io_odom = ct.NonlinearIOSystem(auto_odom_full, None, inputs=('wl', 'wr'), states=('x', 'y', 'th', 'v', 'w'), outputs=('x', 'y', 'th', 'v', 'w'), name='odom', params={'dt': dt})
    eqpt = ct.find_eqpt(io_odom, [x0, y0, theta0, v0, w0], [*action], return_result=return_result)
    print(eqpt)
    xeq = eqpt[0]
    lin_odom = ct.linearize(io_odom, xeq, 0)
    x = lin_odom.A@[[xe] for xe in xeq] + lin_odom.B@[[a] for a in action]
    return x
    