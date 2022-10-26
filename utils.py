from typing import Tuple

import casadi as ca
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
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

def calibration_params(traj, img):
    """
    Get the calibration parameters to fit the simulation to the real track.

    :param traj: the trajectory from the simulation
    :param img: the image from the watchtower

    :return: the calibrated parameters
    """
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 90  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    
    cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range for yellow
    hsv_color1 = np.array([35, 70, 210])
    hsv_color2 = np.array([60, 250, 250])

    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
    
    low_threshold = 89
    high_threshold = 80
    edges = cv2.Canny(mask, low_threshold, high_threshold)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    _points = lines.reshape(-1, 2)
    hull = ConvexHull(points=_points)
    
    borders = _points[hull.vertices]
    scale_x = (borders[:,0].max()-borders[:,0].min())/(traj[:,0].max()-traj[:,0].min())
    scale_y = (borders[:,1].max()-borders[:,1].min())/(traj[:,1].max()-traj[:,1].min())
    
    traj_scaled = traj
    traj_scaled[:,0] = traj_scaled[:,0]*scale_x
    traj_scaled[:,1] = traj_scaled[:,1]*scale_y

    offset_x = borders[:,0].min() - traj_scaled[:,0].min()
    offset_y = borders[:,1].min() - traj_scaled[:,1].min()
    
    return scale_x, scale_y, offset_x, offset_y

def casadi_mod(n, base):
    """
    Modulo function for casadi.
    
    :param n: number to be moduloed
    :param base: modulo base
    """
    return n - ca.floor(n/base) * base

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

def get_cars(img):
    """
    Get the car coordinates and orientation from the image.

    :param img: the image from the watchtower

    :return: the car coordinates and orientation
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_color1_blue = np.array([50, 200, 50])
    hsv_color2_blue = np.array([100, 300, 300])
    mask_blue = cv2.inRange(img_hsv, hsv_color1_blue, hsv_color2_blue)

    hsv_color1_pink = np.array([100, 50, 200])
    hsv_color2_pink = np.array([200, 200, 250])
    mask_pink = cv2.inRange(img_hsv, hsv_color1_pink, hsv_color2_pink)
    
    front_coo = np.argwhere(mask_pink==255).mean(axis=0)[::-1]
    back_coo = np.argwhere(mask_blue==255).mean(axis=0)[::-1]
    
    x_center = (front_coo[0] + back_coo[0])/2
    y_center = (front_coo[1] + back_coo[1])/2
    angle = np.arctan2(-front_coo[1]+back_coo[1], -front_coo[0]+back_coo[0])
    
    return x_center, y_center, angle

def get_dimensions(env):
    """
    Get the dimensions of the environment.
    """
    return (env.grid_height*env.road_tile_size, env.grid_width*env.road_tile_size)

def get_angles(x, y, x0=None, y0=None):
    """
    Get the angles of the trajectory.
    
    :param x: x coordinates
    :param y: y coordinates
    :param x0: x coordinate of the origin
    :param y0: y coordinate of the origin
    """

    if x0 is None:
        x0 = np.mean(x)
    if y0 is None:
        y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    return angles

def get_border(traj, distance=0.15):
    """
    Get the border of the trajectory.
    
    :param traj: the trajectory of the yellow line
    :param distance: the distance from the center

    :return: the borders of the track
    """
    borders_inside = []
    borders_outside = []
    xm, ym = traj.mean(0)
    for idx, _ in enumerate(traj[:-1]):
        x0, y0 = traj[idx]
        x1, y1 = traj[idx+1]
        m = (y1 - y0) / (x1 - x0)
        xp, yp = (x1 + x0) / 2, (y1 + y0) / 2
        mp = -1 / m
        kp = yp - mp * xp
        a = 1+mp**2
        b = -2*xp+2*mp*(kp-yp)
        c = -distance**2+(kp-yp)**2+xp**2
        try:
            xs0, xs1 = np.roots([a,b,c])
        except np.linalg.LinAlgError:
            print("No solution")
            continue
        ys0, ys1 = mp*xs0+kp, mp*xs1+kp

        if (xs0-xm)**2+(ys0-ym)**2 < (xs1-xm)**2+(ys1-ym)**2:
            borders_inside.append([xs0, ys0])
            borders_outside.append([xs1, ys1])
        else:
            borders_inside.append([xs1, ys1])
            borders_outside.append([xs0, ys0])
    borders_inside = np.array(borders_inside)
    borders_outside = np.array(borders_outside)
    return borders_inside, borders_outside

def get_casadi_interpolation(env, show_result=False):
    """
    Get the interpolation function of the trajectory of the agent in the environment but for casadi.
    Interpolate on the already done interpolation to keep the advantages of scipy.
    """
    points = get_trajectory(env, samples=200, scaled=True)
    x, y = points.T
    x_sorted, y_sorted, x0, y0 = sort_xy(x, y, return_origin=True)
    angles = get_angles(x_sorted, y_sorted, x0=x0, y0=y0)
    spline_x = ca.interpolant('LUT','bspline', [angles], x_sorted)
    spline_y = ca.interpolant('LUT','bspline', [angles], y_sorted)
    if show_result:
        plt.figure()
        plt.plot(spline_x(angles), spline_y(angles))
        plt.show()
    return spline_x, spline_y, x_sorted, y_sorted, x0, y0

def get_position(env) -> Position:
    """
    Get the position of the agent in the environment.
    """
    p_se_matrix = env.cartesian_from_weird(env.cur_pos, env.cur_angle)
    # p_string = geometry.SE2.friendly(p_se_matrix)
    p = Position(x=p_se_matrix[0, 2], y=p_se_matrix[1, 2], theta=env.cur_angle)
    # p.from_friendly(p_string)
    return p

def get_top_view(env):
    """
    Top down has black borders, cut them and return only the track.

    :param env: the environment
    """
    img = env.render(mode="top_down")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,21,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop

def get_interpolation(env, no_preprocessing=False, return_origin=False, scaled=True, method="distance"):
    """
    Get the interpolation function of the trajectory of the agent in the environment.

    :param no_preprocessing: if True, the trajectory is not preprocessed
    :param return_origin: if True, the origin is returned
    :param scaled: if True, the coordinates are scaled
    :param method: if "angle", the angles are used, if "distance", the distance from starting point is used

    :return: np.array
    """
    env.reset()
    top_view = np.flip(get_top_view(env), [0])
    # top_view = get_top_view(env)

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

    if scaled:
        x, y = image_to_tile_coordinates(x, y, env)

    x_sorted, y_sorted, x0, y0 = sort_xy(x, y, return_origin=True)


    if method == "angle":
        # Interpolation angle-based
        angles = get_angles(x_sorted, y_sorted, x0=x0, y0=y0)
        # Add first and last point
        spline_input = np.concatenate([[0], angles, [2*np.pi]])
        x_sorted = np.concatenate([[x_sorted[-1]], x_sorted, [x_sorted[0]]])
        y_sorted = np.concatenate([[y_sorted[-1]], y_sorted, [y_sorted[0]]])
    elif method == "distance":
        # Interpolation distance-based
        points = np.array([x_sorted, y_sorted]).T
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        spline_input = np.insert(distance, 0, 0)/distance[-1]
    else:
        raise ValueError("Unknown method, must be 'angle' or 'distance'")


    s = 0.006 if scaled else 0.01

    spline_x = UnivariateSpline(spline_input, x_sorted, k=2, s=s)
    spline_y = UnivariateSpline(spline_input, y_sorted, k=2, s=s)

    if return_origin:
        return spline_x, spline_y, x_sorted, y_sorted, x0, y0

    return [spline_x, spline_y]

def get_top_view_shape(env):
    env.reset()
    top_view = get_top_view(env)
    return top_view.shape

def get_trajectory(env, no_preprocessing=False, samples=50, scaled=True, method="distance", last_value_distance=1):
    """
    Get some points from the trajectory of the agent in the environment.

    :param no_preprocessing: if True, the trajectory is not preprocessed
    :param samples: the number of samples to take
    :param scaled: if True, the trajectory is scaled to the environment size
    :param method: if "angle", the angles are used, if "distance", the distance from starting point is used
    :param last_value_distance: if method is "distance", the last value is the scaled distance from the starting point

    :return: np.array
    """
    splines = get_interpolation(env, no_preprocessing=no_preprocessing, scaled=scaled, method=method)

    # Computed the spline for the asked distances:
    if method == "angle":
        alpha = np.linspace(0, 2*np.pi, samples)
    elif method == "distance":
        alpha = np.linspace(0, last_value_distance, samples)
    else:
        raise ValueError("Unknown method, must be 'angle' or 'distance'")
    points_fitted = np.vstack( spl(alpha) for spl in splines ).T

    return points_fitted
        

def image_to_tile_coordinates(x, y, env):
    """
    Convert image coordinates to tile coordinates.
    :param x: x coordinates
    :param y: y coordinates
    :param env: the environment
    
    :return: x, y
    """
    top_x, top_y, z = get_top_view_shape(env)

    x = x*env.grid_width*env.road_tile_size/top_y
    y = y*env.grid_height*env.road_tile_size/top_x
    return x, y

def model_F(dt=0.033):
    """
    Return the model casadi function.

    :param dt: the time step
    """
    up = 5
    wp = 4
    # parameters for forced dynamics
    u_alpha_r = 1.5
    u_alpha_l = 1.5
    w_alpha_r = 15  # modify this for trim
    w_alpha_l = 15
    # States
    x0 = ca.MX.sym('x')
    y0 = ca.MX.sym('y')
    th0 = ca.MX.sym('th')
    w0 = ca.MX.sym('w')
    v0 = ca.MX.sym('v')
    x = ca.vertcat(x0, y0, th0, v0, w0) # Always vertically concatenate the states --> [n_x,1]
    # Inputs
    wl = ca.MX.sym('wl')
    wr = ca.MX.sym('wr')
    u = ca.vertcat(wl, wr) # Always vertically concatenate the inputs --> [n_u,1]
    # System dynamics (CAN BE NONLINEAR! ;))
    # x_long_dot_dot = -u1*v0 + u_alpha_r*wr + u_alpha_l*wl
    # w_dot_dot = -w1*w0 + w_alpha_r*wr - w_alpha_l*wl
    v1 = (1-up*dt)*v0 + u_alpha_r*dt*wr + u_alpha_l*dt*wl
    w1 = (1-wp*dt)*w0 + w_alpha_r*dt*wr - w_alpha_l*dt*wl
    x1 = x0 + v0*dt*np.cos(th0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(th0 + w0*dt/2)
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

def model_F_tuned(dt=0.033, method="speed"):
    """
    Return the model casadi function tuned according to the parameter found in the thesis.

    :param dt: the time step
    """
    if method == "speed":
        u1 = 2.14148837
        u2 = 0.12200042
        u3 = -0.28237442
        w1 = 1.3380637
        w2 = 0.40072379
        w3 = 1.30781483
        u_alpha_r = 1.30781483
        u_alpha_l = 1.03762896
        w_alpha_r = 2.9650673
        w_alpha_l = 2.89169198
    elif method == "position":
        u1 = 4.3123709
        u2 = 0.42117578
        u3 = 0. 
        w1 = 1.34991163
        w2 = 0.66724572
        w3 = 0.74908594
        u_alpha_r = 2.27306332
        u_alpha_l = 0.73258966
        w_alpha_r = 3.12010274
        w_alpha_l = 2.86162447
    else:
        raise ValueError("Unknown method, must be 'speed' or 'position'")
    # States
    x0 = ca.MX.sym('x')
    y0 = ca.MX.sym('y')
    th0 = ca.MX.sym('th')
    w0 = ca.MX.sym('w')
    v0 = ca.MX.sym('v')
    x = ca.vertcat(x0, y0, th0, v0, w0) # Always vertically concatenate the states --> [n_x,1]
    # Inputs
    wl = ca.MX.sym('wl')
    wr = ca.MX.sym('wr')
    u = ca.vertcat(wl, wr) # Always vertically concatenate the inputs --> [n_u,1]
    
    # V =  [[wl], [wr]]
    V = ca.vertcat(wl, wr)

    f_dynamic = ca.vertcat(-u1 * v0 - u2 * w0 + u3 * w0 ** 2, -w1 * w0 - w2 * v0 - w3 * v0 * w0)
    # input Matrix
    B = ca.DM([[u_alpha_r, u_alpha_l], [w_alpha_r, -w_alpha_l]])
    # forced response
    f_forced = B@V
    # acceleration
    X_dot_dot = f_dynamic + f_forced
    
    v1 = v0 + X_dot_dot[0] * dt
    w1 = w0 + X_dot_dot[1] * dt
    x1 = x0 + v0*dt*np.cos(th0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(th0 + w0*dt/2)
    # Cannot use atan2 because x1 and y1 are approximated while th1 is not
    theta1 = th0 + w0*dt
    dae = ca.vertcat(x1, y1, theta1, v1, w1)
    F = ca.Function('F',[x,u],[dae],['x','u'],['dae'])
    return F

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

def show_on_map(env, poses: Position, fig=plt):
    """
    Show the pose on the map.

    :param env: the environment
    :param pose: list of points of type Position
    :param fig: the figure to plot on
    """
    env.reset()
    top_view = get_top_view(env)
    fig.plot([p.x*top_view.shape[1]/(env.grid_width*env.road_tile_size) for p in poses], [p.y*top_view.shape[0]/(env.grid_height*env.road_tile_size) for p in poses], c='r')
    fig.imshow(top_view, origin='lower')

def show_on_map_array(env, poses, top_view=None, fig=plt):
    """
    Show the pose on the map.

    :param env: the environment
    :param pose: list of points [[x0, y0], [x1, y1], ...]
    :param top_view: the top view image, if None, it is computed
    :param fig: the figure to plot on
    """
    if top_view == None:
        top_view = get_top_view(env)
    fig.plot(poses[:, 0]*top_view.shape[1]/(env.grid_width*env.road_tile_size), poses[:, 1]*top_view.shape[0]/(env.grid_height*env.road_tile_size), c='r')
    fig.imshow(top_view, origin='lower')

def sort_xy(x, y, return_origin=False):
    """
    Sort by angle

    :param return_origin: If true returns also the computed origin
    """

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    if return_origin:
        return x_sorted, y_sorted, x0, y0

    return x_sorted, y_sorted
    
def use_calibration(traj, scale_x=None, scale_y=None, offset_x=None, offset_y=None, img=None):
    """
    Use the calibration to transform the trajectory.

    :param traj: the trajectory from the simulation
    :param scale_x: the x scale
    :param scale_y: the y scale
    :param offset_x: the x offset
    :param offset_y: the y offset
    :param img: the image from the watchtower

    :return: the transformed trajectory
    """
    if scale_x == None or scale_y == None or offset_x == None or offset_y == None:
        if img is None:
            raise ValueError("Please provide the calibration values scale_x, scale_y, offset_x, offset_y or the img.")
        scale_x, scale_y, offset_x, offset_y = calibration_params(traj, img)
    traj[:,0] = traj[:,0]*scale_x + offset_x
    traj[:,1] = traj[:,1]*scale_y + offset_y
    return traj