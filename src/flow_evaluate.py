import csv
import os

import numpy as np

import image_io
import flow_io
import flow_postproc
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
        log.debug("In the optical flow there are {} ({}) pixels are available.".format(np.sum(index_valid), np.sum(index_valid) / float(index_valid.size)))

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


def opticalflow_metric(flo_gt, flo_eva, erp_flo_error_filename_prefix=None, flo_mask=None, min_ratio=0.1, max_ratio=0.9, verbose=True):
    """ Compute error and error map.

    :param flo_gt: The ground truth optical flow.
    :type flo_gt: numpy
    :param flo_eva: The evaluated optical flow.
    :type flo_eva: numpy
    :param erp_flo_error_filename_prefix: The outo
    :type erp_flo_error_filename_prefix: str
    :param flo_mask: The optical flow mask, 0 is unavailable, defaults to None
    :type flo_mask: numpy, optional
    """
    # error value
    aae = AAE(flo_gt, flo_eva, spherical=False, of_mask=flo_mask)
    epe = EPE(flo_gt, flo_eva, spherical=False, of_mask=flo_mask)
    rms = RMSE(flo_gt, flo_eva, spherical=False, of_mask=flo_mask)
    aae_sph = AAE(flo_gt, flo_eva, spherical=True, of_mask=flo_mask)
    epe_sph = EPE(flo_gt, flo_eva, spherical=True, of_mask=flo_mask)
    rms_sph = RMSE(flo_gt, flo_eva, spherical=True, of_mask=flo_mask)

    if verbose:
        log.debug("AAE: {}".format(aae))
        log.debug("EPE: {}".format(epe))
        log.debug("RMS: {}".format(rms))
        log.debug("AAE Spherical: {}".format(aae_sph))
        log.debug("EPE Spherical: {}".format(epe_sph))
        log.debug("RMS Spherical: {}".format(rms_sph))

    # error map
    if erp_flo_error_filename_prefix is not None:
        log.debug("AAE_mat: {}".format(erp_flo_error_filename_prefix + "_aae_mat.jpg"))
        aae_mat, _ = AAE_mat(flo_gt, flo_eva, False, flo_mask)
        aae_mat_vis = image_io.visual_data(aae_mat, min_ratio, max_ratio)
        image_io.image_save(aae_mat_vis, erp_flo_error_filename_prefix + "_aae_mat.jpg")

        log.debug("AAE_Sph_mat: {}".format(erp_flo_error_filename_prefix + "_aae_mat_sph.jpg"))
        aae_mat_sph, _ = AAE_mat(flo_gt, flo_eva, True, flo_mask)
        aae_mat_sph_vis = image_io.visual_data(aae_mat_sph, min_ratio, max_ratio)
        image_io.image_save(aae_mat_sph_vis, erp_flo_error_filename_prefix + "_aae_mat_sph.jpg")

        log.debug("EPE_mat: {}".format(erp_flo_error_filename_prefix + "_epe_mat.jpg"))
        epe_mat, _ = EPE_mat(flo_gt, flo_eva,  False, flo_mask)
        epe_mat_vis = image_io.visual_data(epe_mat, min_ratio, max_ratio)
        image_io.image_save(epe_mat_vis, erp_flo_error_filename_prefix + "_epe_mat.jpg")

        log.debug("EPE_Sph_mat: {}".format(erp_flo_error_filename_prefix + "_epe_mat_sph.jpg"))
        epe_mat_sph, _ = EPE_mat(flo_gt, flo_eva,  True, flo_mask)
        epe_mat_sph_vis = image_io.visual_data(epe_mat_sph, min_ratio, max_ratio)
        image_io.image_save(epe_mat_sph_vis, erp_flo_error_filename_prefix + "_epe_mat_sph.jpg")

        log.debug("RMS_mat: {}".format(erp_flo_error_filename_prefix + "_rms_mat.jpg"))
        rms_mat, _ = RMSE_mat(flo_gt, flo_eva,  False, flo_mask)
        rms_mat_vis = image_io.visual_data(rms_mat, min_ratio, max_ratio)
        image_io.image_save(rms_mat_vis, erp_flo_error_filename_prefix + "_rms_mat.jpg")

        log.debug("RMS_Sph_mat: {}".format(erp_flo_error_filename_prefix + "_rms_mat_sph.jpg"))
        rms_mat_sph, _ = RMSE_mat(flo_gt, flo_eva,  True, flo_mask)
        rms_mat_sph_vis = image_io.visual_data(rms_mat_sph, min_ratio, max_ratio)
        image_io.image_save(rms_mat_sph_vis, erp_flo_error_filename_prefix + "_rms_mat_sph.jpg")

    return aae, epe, rms, aae_sph, epe_sph, rms_sph


def opticalflow_metric_folder(of_dir, of_gt_dir, mask_filename_exp=None, result_csv_filename=None, visual_of_error=False, of_wraparound=True, skip_list = []):
    """
    1) "AAE", 'EPE', 'RMS', "SAAE", "SEPE", "SRMS" error to csv 
    2) output optical flow error visualized image

    :param of_dir: The absolute path of optical flow folder.
    :type of_dir: str
    :param of_gt_dir: The absolute path of ground truth optical flow folder.
    :type of_gt_dir: str
    :param mask_filename_exp: The mask file name expression.
    :type mask_filename_exp: str
    :param result_csv_filename: the csv file name , output to of_dir, defaults to None
    :type result_csv_filename: str, optional
    :param visual_of_error: Whether outout the error map to image, defaults to False,
    :type visual_of_error: bool, optional
    :param of_wraparound: process the wrap around before metric the optical flow.
    :type of_wraparound: bool
    """
    if result_csv_filename is None:
        result_csv_filepath = of_dir + "result.csv"
    else:
        result_csv_filepath = of_dir + result_csv_filename
    result_file = open(result_csv_filepath, mode='w', newline='', encoding='utf-8')
    result_csv_file = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    result_csv_file.writerow(["flo_filename", "AAE", 'EPE', 'RMS', "SAAE", "SEPE", "SRMS"])

    filelist = os.listdir(of_dir)
    filelist.sort()
    for filename in filelist:
        if not filename.endswith(".flo"):
            continue
        if filename in skip_list:
            print("skip {} in {}".format(filename, skip_list))
            continue

        # load of and gt of
        optical_flow = flow_io.read_flow_flo(of_dir + filename)
        optical_flow_gt = flow_io.read_flow_flo(of_gt_dir + filename)

        if optical_flow.shape != optical_flow_gt.shape:
            # resize optical flow
            log.warn("The optical flow shape is different, {} and {}.".format(optical_flow.shape, optical_flow_gt.shape))
            width = optical_flow_gt.shape[1]
            height = optical_flow_gt.shape[0]
            optical_flow = flow_postproc.flow_resize(optical_flow, width_new=width, height_new=height)

        if of_wraparound:
            optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
            optical_flow_gt = flow_postproc.erp_of_wraparound(optical_flow_gt)
        mask_erp_image = None
        if mask_filename_exp is not None:
            image_index = filename[0:4]
            mask_erp_image = image_io.image_read(of_gt_dir + mask_filename_exp.format(int(image_index)))

        # metric error
        error_mat_vis_min_ratio = 0.1
        error_mat_vis_max_ratio = 0.9
        if visual_of_error:
            aae, epe, rms, aae_sph, epe_sph, rms_sph = \
                opticalflow_metric(optical_flow_gt, optical_flow, erp_flo_error_filename_prefix=of_dir + filename, flo_mask=mask_erp_image,
                                   min_ratio=error_mat_vis_min_ratio, max_ratio=error_mat_vis_max_ratio, verbose=True)
        else:
            aae, epe, rms, aae_sph, epe_sph, rms_sph = \
                opticalflow_metric(optical_flow_gt, optical_flow, erp_flo_error_filename_prefix=None, flo_mask=mask_erp_image,
                                   min_ratio=error_mat_vis_min_ratio, max_ratio=error_mat_vis_max_ratio, verbose=True)
        result_csv_file.writerow([filename, aae, epe, rms, aae_sph, epe_sph, rms_sph])
        result_file.flush()

    result_file.close()
