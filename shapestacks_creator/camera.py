from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax = Axes3D(fig)
PI = np.pi
EPS = np.finfo(float).eps * 4.

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.:
        quat[3] = 1.
    elif quat[3] < -1.:
        quat[3] = -1.

    den = np.sqrt(1. - quat[3] * quat[3])
    if math.isclose(den, 0.):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2. * math.acos(quat[3])) / den

# @jit_decorator
def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (np.array): 3x3 rotation matrix
    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def vec(values):
    """
    Converts value tuple into a numpy vector.
    Args:
        values (n-array): a tuple of numbers
    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)

def get_pose_matrix(x: float=0, y: float=0, z: float=0,
                    phi: float=0,  theta: float=0, psi: float=0) -> np.array:
    """
    Compute pose matrix for given translation/rotation parameters
    Parameters
    ----------
    x : float, optional
        x coordinate. The default is 0.
    y : float, optional
        y coordinate. The default is 0.
    z : float, optional
        z coordinate. The default is 0.
    phi : float, optional
        rotation around x axis in degrees. The default is 0.
    theta : float, optional
        rotation around y axis in degrees. The default is 0.
    psi : float, optional
        rotation around x axis in degree. The default is 0.
    Returns
    -------
    pose : np.array (4, 4)
        pose matrix in homogeneous representation.
    """
    rot = R.from_euler('xyz', [phi, theta, psi], degrees=True).as_matrix()
    trans = np.array([[x, y, z]])
    pose = np.concatenate((np.concatenate((rot, trans.T), axis=1),
                           [[0, 0, 0, 1]]), axis=0)
    return pose


def get_xyzphitheta(pose: np.array) -> np.array:
    """
    Computes the vector (x, y, z, phi, theta) given a pose matrix
    Parameters
    ----------
    pose : np.array (4, 4)
        pose matrix in homogeneous representation.
    Returns
    -------
    xyzphitheta : np.array (5, )
        camera transform vector
    """
    trans = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3])
    phi, theta, psi = rot.as_euler('xyz', degrees=True)
    xyzphitheta = np.concatenate((trans, [-phi, theta, psi]))
    return xyzphitheta


def get_circle_pose(theta: float, r: float) -> np.array:
    """
    Compute pose matrix for angle theta in xz-circle with radius r around
    y-axis (theta = 0 corresponds position (0, 0, r))
    Parameters
    ----------
    theta : float
        rotation around y axis in degrees.
    r : float
        radius of circle.
    Returns
    -------
    pose : np.array (4, 4)
        pose matrix in homogeneous representation.
    """
    z = r*np.cos(np.radians(theta))
    x = r*np.sin(np.radians(theta))
    pose = get_pose_matrix(x=x, z=z, theta=theta)
    return pose


def get_sphere_pose(phi: float, theta: float, r: float) -> np.array:
    """
    Compute pose matrix for angle theta in xz-circle with radius r around
    y-axis and angle phi in yz-circle around x-axis (spherical coordinates)
    Parameters
    ----------
    phi : float
        rotation around x axis in degrees.
    theta : float
        rotation around y axis in degrees.
    r : float
        radius of circle.
    Returns
    -------
    pose : np.array (4, 4)
        pose matrix in homogeneous representation.
    """
    z = r*np.cos(np.radians(phi))*np.cos(np.radians(theta))
    x = r*np.cos(np.radians(phi))*np.sin(np.radians(theta))
    y = r*np.sin(np.radians(phi))
    pose = get_pose_matrix(x=x, y=y, z=z, theta=theta, phi=-phi)
    return pose


def get_sphere_poses(start_angle: float, end_angle: float,
                     number_steps: int, r: float) -> np.array:
    """
    Compute poses on a sphere between start and end angle (for phi, theta)
    Parameters
    ----------
    start_angle : float
        start angle for theta and phi in degrees.
    end_angle : float
        end angle for theta and phi in degrees.
    number_steps : int
        number of steps between start and end angle.
    r : float
        radius of sphere.
    Returns
    -------
    poses : np.array (number_steps ** 2, 4, 4)
        pose matrices in homogeneous representation.
    """
    phis = np.linspace(start_angle, end_angle, number_steps)
    print("Angle stepsize: {:.2f}°".format((end_angle - start_angle)/number_steps))
    thetas = np.linspace(start_angle, end_angle, number_steps)
    angles = np.transpose([np.tile(phis, len(thetas)),
                           np.repeat(thetas, len(phis))])
    poses = [get_sphere_pose(phi, theta, r) for (phi, theta) in angles]
    return np.array(poses), angles


def get_circle_poses(start_angle: float, end_angle: float,
                     number_steps: int, r: float) -> np.array:
    """
    Compute poses on a circle between start and end angle (for theta)
    Parameters
    ----------
    start_angle : float
        start angle for theta in degrees.
    end_angle : float
        end angle for theta in degrees.
    number_steps : int
        number of steps between start and end angle.
    r : float
        radius of circle.
    Returns
    -------
    poses : np.array (number_steps, 4, 4)
        pose matrices in homogeneous representation.
    """
    print("Angle stepsize: {:.2f}°".format((end_angle - start_angle)/number_steps))
    thetas = np.linspace(start_angle, end_angle, number_steps)
    poses = [get_circle_pose(theta, r) for theta in thetas]
    return np.array(poses), thetas


def get_circle_on_sphere_poses(number_steps: int, circle_radius: float,
                               sphere_radius: float, center_theta: float=0, 
                               center_phi: float=0) -> np.array:
    """
    Compute poses on a circle with radius circle_radius on a sphere with
    radius sphere_radius
    Parameters
    ----------
    number_steps : int
        number of steps inbetween the circle.
    circle_radius : float
        radius of circle.
    sphere_radius : float
        radius of sphere.
    center_theta: float
        theta of center of circle on sphere
    center_phi: float
        phi of center of circle on sphere
    Returns
    -------
    poses : np.array (number_steps, 4, 4)
        pose matrices in homogeneous representation.
    """
    angles = np.linspace(0, np.pi*2, number_steps)
    print("Angle stepsize: {:.2f}°".format(360/number_steps))
    poses = []
    for angle in angles:
        phi = circle_radius*np.cos(angle) + center_phi
        theta = circle_radius*np.sin(angle) + center_theta
        camera_pose = get_sphere_pose(phi, theta, sphere_radius)
        poses.append(camera_pose)
    return np.array(poses), angles

def camera_origin_direction(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Calculate phi and theta in degrees for a camera to face the origin
    of the coordinate system
    Parameters
    ----------
    x : float
        x coordinate of camera.
    y : float
        y coordinate of camera.
    z : float
        z coordinate of camera.
    Returns
    -------
    Tuple[float, float]
        phi and theta in degrees.
    """
    phi = np.degrees(np.arctan2(y, z))
    theta = np.degrees(np.arctan2(x, z))
    return phi.item(), theta.item()

def get_pose_matrix(x: float=0, y: float=0, z: float=0,
                    phi: float=0,  theta: float=0, psi: float=0) -> np.array:
    """
    Compute pose matrix for given translation/rotation parameters
    Parameters
    ----------
    x : float, optional
        x coordinate. The default is 0.
    y : float, optional
        y coordinate. The default is 0.
    z : float, optional
        z coordinate. The default is 0.
    phi : float, optional
        rotation around x axis in degrees. The default is 0.
    theta : float, optional
        rotation around y axis in degrees. The default is 0.
    psi : float, optional
        rotation around x axis in degree. The default is 0.
    Returns
    -------
    pose : np.array (4, 4)
        pose matrix in homogeneous representation.
    """
    rot = R.from_euler('xyz', [phi, theta, psi], degrees=True).as_matrix()
    trans = np.array([[x, y, z]])
    pose = np.concatenate((np.concatenate((rot, trans.T), axis=1),
                           [[0, 0, 0, 1]]), axis=0)
    return pose


def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to euler angles in radian.
    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)
    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))

# poses, angles = get_sphere_poses(0, 360, 50, 1.0)
# pos = poses[:, :3, 3]
# ind = np.where(pos[:, 2]>0.1)[0]
# x = pos[ind, 0]
# y = pos[ind, 1]
# z = pos[ind, 2]
#
#
# ax.scatter(x,y,z, marker='o', s=20, c="goldenrod", alpha=0.6)
# plt.show()
# # for ii in range(0,360,1):
# #     ax.view_init(elev=10., azim=ii)
# #     plt.savefig("movie%d.png" % ii)
