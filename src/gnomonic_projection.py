import numpy as np
from scipy.spatial.transform import Rotation as R

import spherical_coordinates

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Implement the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""


def gnomonic_projection(theta, phi, theta_0, phi_0, label_outrange_pixel=False):
    """ Gnomonic projection.
    Convet point form the spherical coordinate to tangent image's coordinate.
        https://mathworld.wolfram.com/GnomonicProjection.html

    :param theta: spherical coordinate's longitude.
    :type theta: numpy
    :param phi: spherical coordinate's latitude.
    :type phi: numpy
    :param theta_0: the tangent point's longitude of gnomonic projection.
    :type theta_0: float
    :param phi_0: the tangent point's latitude of gnomonic projection.
    :type phi_0: float
    :param label_outrange_pixel: If True set the outrange to Nan, if False do nothing.
    :type label_outrange_pixel: bool, optional
    :return: The gnomonic coordinate normalized coordinate.
    :rtype: tuple
    """
    cos_c = np.sin(phi_0) * np.sin(phi) + np.cos(phi_0) * np.cos(phi) * np.cos(theta - theta_0)

    # get cos_c's zero element index
    zeros_index = cos_c == 0
    if np.any(zeros_index):
        cos_c[zeros_index] = np.finfo(np.float).eps

    x = np.cos(phi) * np.sin(theta - theta_0) / cos_c
    y = (np.cos(phi_0) * np.sin(phi) - np.sin(phi_0) * np.cos(phi) * np.cos(theta - theta_0)) / cos_c

    if np.any(zeros_index):
        x[zeros_index] = 0
        y[zeros_index] = 0

    # check if the points on the hemisphere of tangent point
    if label_outrange_pixel:
        dist_array = spherical_coordinates.great_circle_distance_uv(theta, phi, theta_0, phi_0)
        overflow_point = dist_array >= (np.pi * 0.5)
        return x, y, overflow_point
    else:
        return x, y, None


def reverse_gnomonic_projection(x, y, theta_0, phi_0, hemisphere_index=None):
    """ Reverse gnomonic projection.
    Convert the gnomonic nomalized coordinate to spherical coordinate.

    :param x: the gnomonic plane coordinate x.
    :type x: numpy 
    :param y: the gnomonic plane coordinate y.
    :type y: numpy
    :param theta_0: the gnomonic projection tangent point's longitude.
    :type theta_0: float
    :param phi_0: the gnomonic projection tangent point's latitude f .
    :type phi_0: float
    :param hemisphere_index: the index of hemisphere, true in the hemisphere same with the tangent image, false is on the another one.
    :type hemisphere_index: numpy
    :return: the point array's spherical coordinate location. the longitude range is continuous and exceed the range [-pi, +pi]
    :rtype: numpy
    """
    rho = np.sqrt(x**2 + y**2)

    # get rho's zero element index
    zeros_index = rho == 0
    if np.any(zeros_index):
        rho[zeros_index] = np.finfo(np.float).eps

    c = np.arctan2(rho, 1)
    phi_ = np.arcsin(np.cos(c) * np.sin(phi_0) + (y * np.sin(c) * np.cos(phi_0)) / rho)
    theta_ = theta_0 + np.arctan2(x * np.sin(c), rho * np.cos(phi_0) * np.cos(c) - y * np.sin(phi_0) * np.sin(c))

    if hemisphere_index is not None:
        need_around = ~hemisphere_index
        theta_[need_around], phi_[need_around] = spherical_coordinates.sph_coord_modulo(theta_[need_around] + np.pi, -phi_[need_around])

    if np.any(zeros_index):
        phi_[zeros_index] = phi_0
        theta_[zeros_index] = theta_0

    return theta_, phi_


def gnomonic2pixel(coord_gnom_x, coord_gnom_y,
                   padding_size,
                   tangent_image_width, tangent_image_height=None,
                   coord_gnom_xy_range=None):
    """Transform the tangent image's gnomonic coordinate to tangent image pixel coordinate.

    The tangent image gnomonic x is right, y is up.
    The tangent image pixel coordinate is x is right, y is down.

    :param coord_gnom_x: tangent image's normalized x coordinate
    :type coord_gnom_x: numpy
    :param coord_gnom_y: tangent image's normalized y coordinate
    :type coord_gnom_y: numpy
    :param padding_size: in gnomonic coordinate system, padding outside to boundary
    :type padding_size: float
    :param tangent_image_width: the image width with padding
    :type tangent_image_width: float
    :param tangent_image_height: the image height with padding
    :type tangent_image_height: float
    :param coord_gnom_xy_range: the range of gnomonic coordinate, [x_min, x_max, y_min, y_max]
    :type coord_gnom_xy_range: list
    :retrun: the pixel's location 
    :rtype: numpy (int)
    """
    if padding_size != 0:
        log.warn("The padding size is not 0, please check if coord_gnom_xy_range has included the padding!")

    if tangent_image_height is None:
        tangent_image_height = tangent_image_width

    # the gnomonic coordinate range of tangent image
    if coord_gnom_xy_range is None:
        x_min = -1.0
        x_max = 1.0
        y_min = -1.0
        y_max = 1.0
    else:
        x_min = coord_gnom_xy_range[0]
        x_max = coord_gnom_xy_range[1]
        y_min = coord_gnom_xy_range[2]
        y_max = coord_gnom_xy_range[3]

    # normailzed tangent image space --> tangent image space
    gnomonic2image_width_ratio = (tangent_image_width - 1.0) / (x_max - x_min + padding_size * 2.0)
    coord_pixel_x = (coord_gnom_x - x_min + padding_size) * gnomonic2image_width_ratio
    coord_pixel_x = (coord_pixel_x + 0.5).astype(np.int)

    gnomonic2image_height_ratio = (tangent_image_height - 1.0) / (y_max - y_min + padding_size * 2.0)
    coord_pixel_y = -(coord_gnom_y - y_max - padding_size) * gnomonic2image_height_ratio
    coord_pixel_y = (coord_pixel_y + 0.5).astype(np.int)

    return coord_pixel_x, coord_pixel_y


def pixel2gnomonic(coord_pixel_x, coord_pixel_y,  padding_size,
                   tangent_image_width, tangent_image_height=None,
                   coord_gnom_xy_range=None):
    """Transform the tangent image's from tangent image pixel coordinate to gnomonic coordinate.

    :param coord_pixel_x: tangent image's pixels x coordinate
    :type coord_pixel_x: numpy
    :param coord_pixel_y: tangent image's pixels y coordinate
    :type coord_pixel_y: numpy
    :param padding_size: in gnomonic coordinate system, padding outside to boundary
    :type padding_size: float
    :param tangent_image_width: the image size with padding
    :type tangent_image_width: numpy
    :param tangent_image_height: the image size with padding
    :type tangent_image_height: numpy
    :param coord_gnom_xy_range: the range of gnomonic coordinate, [x_min, x_max, y_min, y_max]. It desn't includes padding outside to boundary.
    :type coord_gnom_xy_range: list
    :retrun: the pixel's location 
    :rtype:
    """
    if tangent_image_height is None:
        tangent_image_height = tangent_image_width

    # the gnomonic coordinate range of tangent image
    if coord_gnom_xy_range is None:
        x_min = -1.0
        x_max = 1.0
        y_min = -1.0
        y_max = 1.0
    else:
        x_min = coord_gnom_xy_range[0]
        x_max = coord_gnom_xy_range[1]
        y_min = coord_gnom_xy_range[2]
        y_max = coord_gnom_xy_range[3]

    # tangent image space --> tangent normalized space
    gnomonic_size_x = abs(x_max - x_min)
    gnomonic2image_ratio_width = (tangent_image_width - 1.0) / (gnomonic_size_x + padding_size * 2.0)
    coord_gnom_x = coord_pixel_x / gnomonic2image_ratio_width + x_min - padding_size

    gnomonic_size_y = abs(y_max - y_min)
    gnomonic2image_ratio_height = (tangent_image_height - 1.0) / (gnomonic_size_y + padding_size * 2.0)
    coord_gnom_y = - coord_pixel_y / gnomonic2image_ratio_height + y_max + padding_size

    return coord_gnom_x, coord_gnom_y
