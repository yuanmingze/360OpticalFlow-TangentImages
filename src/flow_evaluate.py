import numpy as np


import flow_warp
import spherical_coordinates

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False

"""
functions used to evaluate the quality of optical flow;
"""

UNKNOWN_FLOW_THRESH = 1e7


def available_pixel(flow, of_mask=None, unknown_value=UNKNOWN_FLOW_THRESH):
    """
    The criterion of the available optical flow pixel.
    1) not very large; 2) not a NaN; 3) mask is not 0;

    :param flow: The optical flow, size is [height, width, 2], U and V.
    :type flow: numpy
    :param of_mask: 0 is the un-available pixel, not 0 is available pixel, size is [height, width].
    :type of_mask: numpy, optional
    :return: available pixel lable matrix, True is valid and False is invalid
    :rtype: numpy, boolon
    """
    min_value = 0
    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]

    # very large optical flow
    index_unlarge = np.logical_not(abs(flow_u) > unknown_value) | (abs(flow_v) > unknown_value)
    # nan optical flow
    index_unnan = np.logical_not(np.isnan(flow_u) | np.isnan(flow_v))
    # zero optical flow
    index_unzero = (np.absolute(flow_u) > min_value) | (np.absolute(flow_v) > min_value)

    # valid pixels index
    index_valid = index_unlarge & index_unnan & index_unzero
    if of_mask is not None:
        index_valid = index_valid & of_mask

    if np.sum(index_valid) != index_valid.size:
        log.info("In the optical flow there are {} pixels are unavailable.".format(np.sum(index_valid)))

    return index_valid


def EPE(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    Endpoint error (EE)
    reference : https://github.com/prgumd/GapFlyt/blob/master/Code/flownet2-tf-umd/src/flowlib.py

    :param of_ground_truth: The ground truth optical flow, [height, width, 2], first channel is U, second channel is U.
    :type of_ground_truth: numpy
    :param of_evaluation: @see of_ground_truth
    :type of_evaluation: numpy
    :param spherical: True: the EPE is geodesic distance of united sphere, False: EPE is image pixels distance.
    :type spherical: bool, optional
    :return: average of EPE
    :rtype: float
    """
    epe, of_gt_available_index = EPE_mat(of_ground_truth, of_evaluation, spherical, of_mask)
    mepe = np.sum(epe) / np.sum(of_gt_available_index)
    return mepe


def EPE_mat(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    @see EPE

    :return: the each pixel is EPE, unavailable pixels is 0. [height, width]
    :rtype: numpy
    """
    of_gt_u = of_ground_truth[:, :, 0]
    of_gt_v = of_ground_truth[:, :, 1]
    of_u = of_evaluation[:, :, 0]
    of_v = of_evaluation[:, :, 1]

    # set the invalid optical flow
    of_gt_available_index = available_pixel(of_ground_truth, of_mask)
    of_unavailable_index = np.logical_not(of_gt_available_index)
    of_gt_u[of_unavailable_index] = 0
    of_gt_v[of_unavailable_index] = 0
    of_u[of_unavailable_index] = 0
    of_v[of_unavailable_index] = 0

    if spherical:
        # compute the end point
        of_gt_endpoints = flow_warp.flow_warp_meshgrid(of_gt_u, of_gt_v)
        of_gt_endpoints_uv = spherical_coordinates.erp2sph(of_gt_endpoints)

        of_eva_endpoints = flow_warp.flow_warp_meshgrid(of_u, of_v)
        of_eva_endpoints_uv = spherical_coordinates.erp2sph(of_eva_endpoints)

        # get great circle distance
        epe = spherical_coordinates.great_circle_distance(of_gt_endpoints_uv, of_eva_endpoints_uv)
    else:
        epe = np.sqrt((of_gt_u - of_u) ** 2 + (of_gt_v - of_v) ** 2)

    return epe, of_gt_available_index


def RMSE(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    compute the root mean square error(RMSE) of optical flow.
    @see EPE

    :retrun: RMSE
    :rtype: float
    """
    rmse_mat, of_gt_available_index = RMSE_mat(of_ground_truth, of_evaluation, spherical, of_mask)
    rmse = np.sqrt(np.sum(np.square(rmse_mat)) / np.sum(of_gt_available_index))
    return rmse


def RMSE_mat(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    Compute pixel-wise ((u_e^i - u_g^i)^2 + (v_e^i - v_g^i)^2).
    @see EPE

    :retrun: each pixel square distance
    :rtype: numpy
    """
    of_gt_u = of_ground_truth[:, :, 0]
    of_gt_v = of_ground_truth[:, :, 1]
    of_u = of_evaluation[:, :, 0]
    of_v = of_evaluation[:, :, 1]

    # ignore the invalid data
    of_gt_available_index = available_pixel(of_ground_truth, of_mask)
    of_unavailable_index = np.logical_not(of_gt_available_index)
    of_gt_u[of_unavailable_index] = 0
    of_gt_v[of_unavailable_index] = 0
    of_u[of_unavailable_index] = 0
    of_v[of_unavailable_index] = 0

    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = flow_warp.flow_warp_meshgrid(of_gt_u, of_gt_v)
        of_gt_endpoints_uv = spherical_coordinates.erp2sph(of_gt_endpoints)
        of_eva_endpoints = flow_warp.flow_warp_meshgrid(of_u, of_v)
        of_eva_endpoints_uv = spherical_coordinates.erp2sph(of_eva_endpoints)

        # get the Spherical Triangle angle
        rmse_mat = spherical_coordinates.great_circle_distance(of_gt_endpoints_uv, of_eva_endpoints_uv)
        # rmse_mat = rmse_mat ** 2
    else:
        diff_u = of_gt_u - of_u
        diff_v = of_gt_v - of_v
        rmse_mat = np.sqrt(diff_u ** 2 + diff_v ** 2)
    return rmse_mat, of_gt_available_index


def AAE(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    The average angular error(AAE) 
    The result is between 0 and 2 * PI.

    :retrun: AAE
    :rtype: float
    """
    aae_mat, of_available_index = AAE_mat(of_ground_truth, of_evaluation, spherical=False, of_mask=None)
    return np.sum(aae_mat) / np.sum(of_available_index)


def AAE_mat(of_ground_truth, of_evaluation, spherical=False, of_mask=None):
    """
    Return the mat of the average angular error(AAE) 
    The angle is radian [0, 2 * PI].

    :retrun: AAE matrix.
    :rtype: numpy 
    """
    of_gt_u = of_ground_truth[:, :, 0]
    of_gt_v = of_ground_truth[:, :, 1]
    of_u = of_evaluation[:, :, 0]
    of_v = of_evaluation[:, :, 1]

    # ignore the invalid data
    of_gt_available_index = available_pixel(of_ground_truth, of_mask)
    of_eva_available_index = available_pixel(of_ground_truth, of_mask)
    of_available_index = of_eva_available_index & of_gt_available_index
    of_unavailable_index = np.logical_not(of_available_index)
    of_gt_u[of_unavailable_index] = 0
    of_gt_v[of_unavailable_index] = 0
    of_u[of_unavailable_index] = 0
    of_v[of_unavailable_index] = 0

    if spherical:
        # get the three points of the triangle
        of_gt_endpoints = flow_warp.flow_warp_meshgrid(of_gt_u, of_gt_v)
        of_gt_endpoints_uv = spherical_coordinates.erp2sph(of_gt_endpoints)
        of_eva_endpoints = flow_warp.flow_warp_meshgrid(of_u, of_v)
        of_eva_endpoints_uv = spherical_coordinates.erp2sph(of_eva_endpoints)
        of_origin_endpoints = flow_warp.flow_warp_meshgrid(np.zeros_like(of_gt_u), np.zeros_like(of_gt_v))
        of_origin_endpoints_uv = spherical_coordinates.erp2sph(of_origin_endpoints)

        # get the Spherical Triangle angle
        angles_mat = spherical_coordinates.get_angle(of_origin_endpoints_uv, of_gt_endpoints_uv, of_eva_endpoints_uv)
    else:
        # compute the average angle
        uv_cross = of_gt_u * of_v - of_gt_v * of_u
        uv_dot = of_gt_u * of_u + of_gt_v * of_v
        angles = np.arctan2(uv_cross, uv_dot)
        angles_mat = angles.reshape(np.shape(of_gt_u))
        angles_mat = abs(angles_mat)

    return angles_mat, of_available_index
