import numpy as np
from scipy.spatial.transform import Rotation as R

import flow_warp

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def great_circle_distance(points_1, points_2, radius=1):
    """
    Get the distance between two points in the sphere, the grate-circle distance.
    Reference: https://en.wikipedia.org/wiki/Great-circle_distance.

    :param points_1: the numpy array [theta_1, phi_1]
    :type points_1: numpy
    :param points_2: the numpy array [theta_2, phi_2]
    :type points_2: numpy
    :param radius: the radius, the default is 1
    :return distance: the great-circle distance of two point.
    :rtype: numpy
    """
    return great_circle_distance_uv(points_1[0], points_1[1], points_2[0], points_2[1], radius)


def great_circle_distance_uv(points_1_theta, points_1_phi, points_2_theta, points_2_phi, radius=1):
    """
    @see great_circle_distance (haversine distances )
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html

    :param points_1_theta: theta in radians, size is [N]
    :type points_1_theta : numpy
    :param points_1_phi: phi in radians, size is [N]
    :type points_1_phi : numpy
    :param points_2_theta: radians
    :type points_2_theta: float
    :param points_2_phi: radians
    :type points_2_phi: float
    :return: The geodestic distance from point ot tangent point.
    :rtype: numpy
    """
    delta_theta = points_2_theta - points_1_theta
    delta_phi = points_2_phi - points_1_phi
    a = np.sin(delta_phi * 0.5) ** 2 + np.cos(points_1_phi) * np.cos(points_2_phi) * np.sin(delta_theta * 0.5) ** 2
    central_angle_delta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    if np.isnan(central_angle_delta).any():
        log.warn("the circle angle have NAN")

    return np.abs(radius * central_angle_delta)


def erp_sph_modulo(theta, phi):
    """Modulo of the spherical coordinate for the erp coordinate.
    
    """
    points_theta = np.remainder(theta + np.pi, 2 * np.pi) - np.pi
    points_phi = -(np.remainder(-phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi)
    return points_theta, points_phi


def erp2sph(erp_points, erp_image_height=None, sph_modulo=False):
    """Convert the point from erp image pixel location to spherical coordinate.
    The image center is spherical coordinate origin.

    :param erp_points: the point location in ERP image x∊[0, width-1], y∊[0, height-1] , size is [2, :]
    :type erp_points: numpy
    :param erp_image_height: ERP image's height, defaults to None
    :type erp_image_height: int, optional
    :param sph_modulo: if true, process the input points wrap around, .
    :type sph_modulo: bool
    :return: the spherical coordinate points, theta is in the range [-pi, +pi), and phi is in the range [-pi/2, pi/2)
    :rtype: numpy
    """
    # 0) the ERP image size
    if erp_image_height == None:
        height = np.shape(erp_points)[1]
        width = np.shape(erp_points)[2]

        if (height * 2) != width:
            log.error("the ERP image width {} is not two time of height {}".format(width, height))
    else:
        height = erp_image_height
        width = height * 2

    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]

    # 1) point location to theta and phi
    points_theta = erp_points_x * (2 * np.pi / width) + np.pi / width - np.pi
    points_phi = -(erp_points_y * (np.pi / height) + np.pi / height * 0.5) + 0.5 * np.pi

    if sph_modulo:
        points_theta, points_phi = erp_sph_modulo(points_theta, points_phi)

    points_theta = np.where(points_theta == np.pi,  -np.pi, points_theta)
    points_phi = np.where(points_phi == -0.5 * np.pi, 0.5 * np.pi, points_phi)

    return np.stack((points_theta, points_phi))


def sph2erp(theta, phi, erp_image_height, sph_modulo=False):
    """ 
    Transform the spherical coordinate location to ERP image pixel location.

    :param theta: longitude is radian
    :type theta: numpy
    :param phi: latitude is radian
    :type phi: numpy
    :param image_height: the height of the ERP image. the image width is 2 times of image height
    :type image_height: [type]
    :param sph_modulo: if yes process the wrap around case, if no do not.
    :type sph_modulo: bool, optional
    :return: the pixel location in the ERP image.
    :rtype: numpy
    """
    if sph_modulo:
        theta, phi = erp_sph_modulo(theta, phi)

    erp_image_width = 2 * erp_image_height
    erp_x = (theta + np.pi) / (2.0 * np.pi / erp_image_width) - 0.5
    erp_y = (-phi + 0.5 * np.pi) / (np.pi / erp_image_height) - 0.5
    return erp_x, erp_y


def car2sph(points_car, min_radius=1e-10):
    """Transform the 3D point from cartesian to unit spherical coordinate.

    :param points_car: The 3D point array, is [point_number, 3], first column is x, second is y, third is z
    :type points_car: numpy
    :param min_radius: The minimized radius.
    :type min_radius: float
    :return: the points spherical coordinate, (theta, phi)
    :rtype: numpy
    """
    radius = np.linalg.norm(points_car, axis=1)

    valid_list = radius > min_radius  # set the 0 radius to origin.

    theta = np.zeros((points_car.shape[0]), np.float)
    theta[valid_list] = np.arctan2(points_car[:, 0][valid_list], points_car[:, 2][valid_list])

    phi = np.zeros((points_car.shape[0]), np.float)
    phi[valid_list] = -np.arcsin(np.divide(points_car[:, 1][valid_list], radius[valid_list]))

    return np.stack((theta, phi), axis=1)


def sph2car(theta, phi, radius=1.0):
    """
    Transform the spherical coordinate to cartesian 3D point.

    :param theta: longitude
    :type theta: numpy
    :param phi: latitude
    :type phi: numpy
    :param radius: the radius of projection sphere
    :type radius: float
    :return: +x right, +y down, +z is froward
    :rtype: numpy
    """
    # points_cartesian_3d = np.array.zeros((theta.shape[0],3),np.float)
    x = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.cos(theta)
    y = -radius * np.sin(phi)

    return np.stack((x, y, z), axis=0)


def rotate_erp_array(erp_image, rotation_mat=None):
    """ Rotate the ERP image with the theta and phi.

    :param erp_image: The ERP image, [height, width, 3]
    :type erp_image: numpy
    :param rotation_mat: The erp rotation matrix.
    :type rotation_mat: numpy
    """
    # flow from tar to src
    opticalflow = rotation2erp_motion_vector(erp_image.shape[0:2], rotation_matrix=rotation_mat.T)
    return flow_warp.warp_backward(erp_image, opticalflow)    # the image backword warp


def rotation2erp_motion_vector(array_size, rotation_matrix=None, wraparound=False):
    """Convert the spherical coordinate rotation to ERP coordinate motion flow.
    With rotate the image's mesh grid.

    :param data_array: the array size, [array_hight, array_width]
    :type data_array: list
    :param rotate_theta: rotate along the longitude, radian
    :type rotate_theta: float
    :param rotate_phi: rotate along the latitude, radian
    :type rotate_phi: float
    """
    # 1) generage spherical coordinate for each pixel
    erp_x = np.linspace(0, array_size[1], array_size[1], endpoint=False)
    erp_y = np.linspace(0, array_size[0], array_size[0], endpoint=False)
    erp_vx, erp_vy = np.meshgrid(erp_x, erp_y)

    # 1) spherical system to Cartesian system and rotate the points
    sph_xy = erp2sph(np.stack((erp_vx, erp_vy)), erp_image_height=array_size[0], sph_modulo=False)
    xyz = sph2car(sph_xy[0], sph_xy[1], radius=1.0)
    xyz_rot = np.dot(rotation_matrix, xyz.reshape((3, -1)))
    array_xy_rot = car2sph(xyz_rot.T).T
    erp_x_rot, erp_y_rot = sph2erp(array_xy_rot[0, :], array_xy_rot[1, :], array_size[0], sph_modulo=False)

    # get motion vector
    motion_vector_x = erp_x_rot.reshape((array_size[0], array_size[1])) - erp_vx
    motion_vector_y = erp_y_rot.reshape((array_size[0], array_size[1])) - erp_vy

    # to wraparound optical flow
    if wraparound:
        cross_minus2pi, cross_plus2pi = sph_wraparound(sph_xy[0], array_xy_rot[0, :].reshape((array_size[0], array_size[1])))
        motion_vector_x[cross_minus2pi] = motion_vector_x[cross_minus2pi] - array_size[1]
        motion_vector_x[cross_plus2pi] = motion_vector_x[cross_plus2pi] + array_size[1]

    return np.stack((motion_vector_x, motion_vector_y), -1)


def rot_sph2mat(theta, phi, degrees_=True):
    """Convert the spherical rotation to rotation matrix.
    """
    return R.from_euler("xyz", [phi, theta, 0], degrees=degrees_).as_matrix()


def rot_mat2sph(rot_mat, degrees_=True):
    """Convert the 3D rotation to spherical coodinate theta and phi.
    """
    euler_angle = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees_)
    return tuple(euler_angle[:2])


def sph_coord_modulo(theta, phi):
    """Modulo the spherical coordinate.
    """
    theta_new = np.remainder(theta + np.pi, np.pi * 2.0) - np.pi
    phi_new = np.remainder(phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    return theta_new, phi_new


def sph_wraparound(src_theta, tar_theta):
    """ Get the line index cross the ERP image boundary.

    :param src_theta: The source point theta.
    :type src_theta: numpy
    :param tar_theta: The target point theta.
    :type tar_theta: numpy
    :return: Pixel index of line wrap around.
    :rtype: numpy
    """
    long_line = np.abs(tar_theta - src_theta) > np.pi
    cross_minus2pi = np.logical_and(src_theta < 0, tar_theta > 0)
    cross_minus2pi = np.logical_and(long_line, cross_minus2pi)
    cross_plus2pi = np.logical_and(src_theta > 0, tar_theta < 0)
    cross_plus2pi = np.logical_and(long_line, cross_plus2pi)

    return cross_minus2pi, cross_plus2pi
