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

    return:
    xdotdot[0] = acc longitudinal
    xdotdot[1] = acc angular
    """
    # https://drive.google.com/file/d/19U1DUo3GtqHxncEKLn2d6RRdLTLgD0Bv/view
    # Paragraph 5.1.5
    
    # wr, wl
    U = np.array([action[1], action[0]])
    V = U.reshape(U.size, 1)
    V = np.clip(V, -1, +1)

    # Previous step linear and angular speed
    # u, w = longit_prev, angular_prev

    ## Calculate Dynamics
    # nonlinear Dynamics - autonomous response
    f_dynamic = np.array([[-u1 * u - u2 * w + u3 * w ** 2], [-w1 * w - w2 * u - w3 * u * w]])  #
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

def get_position(env):
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
        
def next_position(env, dt, omega_l, omega_r) -> Position:
    # Line 2076
    action = DynamicsInfo(motor_left=omega_l, motor_right=omega_r)
    state = env.state.integrate(dt, action)
    q = state.TSE2_from_state()[0]
    q_string = geometry.SE2.friendly(q)
    q_ext = Position()
    q_ext.from_friendly(q_string)
    return q_ext

def my_odometry(action, x0, y0, theta0, v0=0, w0=0, dt=0.033):
    x_dot_dot, w_dot_dot = get_acceleration(action, u=v0, w=w0)

    v1 = v0 + x_dot_dot[0]*dt
    w1 = w0 + w_dot_dot[0]*dt

    # Runge Kutta
    x1 = x0 + v0*dt*np.cos(theta0 + w0*dt/2)
    y1 = y0 + v0*dt*np.sin(theta0 + w0*dt/2)
    theta1 = theta0 + w0*dt

    return Position(x1, y1, theta1), v1, w1

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