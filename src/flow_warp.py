import numpy as np
from scipy import ndimage
from scipy.stats import norm

import pointcloud_utils
import spherical_coordinates
import flow_postproc
import spherical_coordinates as sc

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def flow_warp_meshgrid(motion_flow_u, motion_flow_v):
    """
    warp the the original points (image's mesh grid) with the motion vector, meanwhile process the warp around.

    :param motion_flow_u: [height, width]
    :type motion_flow_u: numpy
    :param motion_flow_v: [height, width]
    :type motion_flow_v: numpy
    :return: the target points
    :rtype: numpy
    """
    if np.shape(motion_flow_u) != np.shape(motion_flow_v):
        log.error("motion flow u shape {} is not equal motion flow v shape {}".format(np.shape(motion_flow_u), np.shape(motion_flow_v)))

    # get the mesh grid
    height = np.shape(motion_flow_u)[0]
    width = np.shape(motion_flow_u)[1]
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    x_array, y_array = np.meshgrid(x_index, y_index)

    # get end point location
    end_points_u = x_array + motion_flow_u
    end_points_v = y_array + motion_flow_v

    # process the warp around
    u_index = end_points_u >= width - 0.5
    end_points_u[u_index] = end_points_u[u_index] - width
    u_index = end_points_u < -0.5
    end_points_u[u_index] = end_points_u[u_index] + width

    v_index = end_points_v >= height-0.5
    end_points_v[v_index] = end_points_v[v_index] - height
    v_index = end_points_v < -0.5
    end_points_v[v_index] = end_points_v[v_index] + height

    return np.stack((end_points_u, end_points_v))


def warp_backward(image_target, of_forward):
    """Backward warp with optical flow from the target image to generate the source image. 

    :param image_target: The terget image of optical flow, [height, width, channel].
    :type image_target: numpy
    :param of_forward:  optical flow from source to target, [height, width, 2].
    :type of_forward: numpy
    :return: Generated source image.
    :rtype: numpy
    """
    image_height = image_target.shape[0]
    image_width = image_target.shape[1]
    image_channels = None
    if len(image_target.shape) == 3:
        image_channels = image_target.shape[2]
    elif len(image_target.shape) == 2:
        image_channels = None
    else:
        log.error("The image shape is {}, do not support.".format(image_target.shape))
    dest_image = np.zeros_like(image_target, dtype=image_target.dtype)

    # 0) comput new location
    x_idx_arr = np.linspace(0, image_width - 1, image_width)
    y_idx_arr = np.linspace(0, image_height - 1, image_height)
    x_idx, y_idx = np.meshgrid(x_idx_arr, y_idx_arr)
    x_idx_new = (x_idx + of_forward[:, :, 0])
    y_idx_new = (y_idx + of_forward[:, :, 1])

    if image_channels is not None:
        for channel_index in range(0, image_channels):
            dest_image[y_idx.astype(int), x_idx.astype(int), channel_index] = ndimage.map_coordinates(image_target[:, :, channel_index], [y_idx_new, x_idx_new], order=1, mode='wrap')
    else:
        dest_image[y_idx.astype(int), x_idx.astype(int)] = ndimage.map_coordinates(image_target[:, :], [y_idx_new, x_idx_new], order=1, mode='constant', cval=255)

    return dest_image


def flow2rotation_3d(erp_flow, mask_method="center"):
    """Compute the two image rotation from the ERP image's optical flow with SVD.
    The rotation is from the first image to second image.

    :param erp_flow: The ERP optical flow. [height, width,2]
    :type erp_flow: numpy
    :param mask_method: center mehtod is just use the point close to obit.
    :type mask_method: str
    :return: The rotation matrix.
    :rtype: numpy
    """
    # 0) source 3D points and target 3D points
    motion_flow_u = erp_flow[:, :, 0]
    motion_flow_v = erp_flow[:, :, 1]
    tar_points_2d = flow_warp_meshgrid(motion_flow_u, motion_flow_v)

    height = np.shape(motion_flow_u)[0]
    width = np.shape(motion_flow_u)[1]
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    x_array, y_array = np.meshgrid(x_index, y_index)
    src_points_2d = np.stack((x_array, y_array))

    # convert to 3D points
    src_points_2d_sph = sc.erp2sph(src_points_2d)
    tar_points_2d_sph = sc.erp2sph(tar_points_2d)

    if mask_method == "center":
        # just use the center rows optical flow
        row_idx_start = int(height * 0.25)
        row_idx_end = int(height * 0.75)
        src_points_2d_sph = src_points_2d_sph[:, row_idx_start:row_idx_end, :]
        tar_points_2d_sph = tar_points_2d_sph[:, row_idx_start:row_idx_end, :]

    src_points_3d = sc.sph2car(src_points_2d_sph[0], src_points_2d_sph[1])
    tar_points_3d = sc.sph2car(tar_points_2d_sph[0], tar_points_2d_sph[1])

    # 1) SVD get the rotation matrix
    src_points_3d = np.swapaxes(src_points_3d.reshape((3, -1)), 0, 1)
    tar_points_3d = np.swapaxes(tar_points_3d.reshape((3, -1)), 0, 1)
    rotation_mat = pointcloud_utils.correpairs2rotation(src_points_3d, tar_points_3d)

    return rotation_mat


def flow2rotation_2d(erp_flow, use_weight=True):
    """Compute the  two image rotation from the ERP image's optical flow.
    The rotation is from the first image to second image.

    :param erp_flow: the erp image's flow 
    :type erp_flow: numpy 
    :param use_weight: use the centre rows and columns to compute the rotation, default is True.
    :type use_weight: bool
    :return: the offset of ERP image, [theta shift, phi shift], radian
    :rtype: float
    """
    erp_image_height = erp_flow.shape[0]
    erp_image_width = erp_flow.shape[1]

    # convert the pixel offset to rotation radian
    erp_flow = flow_postproc.erp_of_wraparound(erp_flow)
    theta_delta_array = 2.0 * np.pi * (erp_flow[:, :, 0] / erp_image_width)
    theta_delta = np.mean(theta_delta_array)

    # just the center column of the optical flow.
    delta = theta_delta / (2.0 * np.pi)
    flow_col_start = int(erp_image_width * (0.5 - delta))
    flow_col_end = int(erp_image_width * (0.5 + delta))
    if delta < 0:
        temp = flow_col_start
        flow_col_start = flow_col_end
        flow_col_end = temp
    flow_col_center = np.full((erp_image_height, erp_image_width), False, dtype=np.bool)
    flow_col_center[:, flow_col_start:flow_col_end] = True
    flow_sign = np.sign(np.sum(np.sign(erp_flow[flow_col_center, 1])))
    if flow_sign < 0:
        positive_index = np.logical_and(erp_flow[:, :, 1] < 0, flow_col_center)
    else:
        positive_index = np.logical_and(erp_flow[:, :, 1] > 0, flow_col_center)
    phi_delta_array = -np.pi * (erp_flow[positive_index, 1] / erp_image_height)

    if use_weight:
        # weight of the u, width
        stdev = erp_image_height * 0.5 * 0.25
        weight_u_array_index = np.arange(erp_image_height)
        weight_u_array = norm.pdf(weight_u_array_index, erp_image_height / 2.0, stdev)
        theta_delta_array = np.average(theta_delta_array, axis=0, weights=weight_u_array)

        # weight of the v, height
        stdev = erp_image_width * 0.5 * 0.25
        weight_v_array_index = np.arange(erp_image_width)
        weight_v_array = norm.pdf(weight_v_array_index, erp_image_width / 2.0, stdev)
        phi_delta_array = np.average(phi_delta_array, axis=1,  weights=weight_v_array)

    phi_delta = np.mean(phi_delta_array)

    return theta_delta, phi_delta


def global_rotation_warping(erp_image, erp_flow, forward_warp=True, rotation_type="3D"):
    """ Global rotation warping.

    Rotate the ERP image base on the flow. 
    If `forward_warp` is True, the `erp_image` is the source image, `erp_flow` is form source to target.
    If `forward_warp` is False, the `erp_image` is the target image, `erp_flow` is from source to target.

    :param erp_image: the image of optical flow, 
    :type erp_image: numpy 
    :param erp_flow: the erp image's flow, from source image to target image.
    :type erp_flow: numpy 
    :param forward_warp: If yes, the erp_image is use the erp_flow forward warp erp_image.
    :type forward_warp: bool 
    :param rotation_type: the global rotation method.
    :type rotation_type: str
    :return: The rotated ERP image,  the rotation matrix from original to target (returned erp image).
    :rtype: numpy
    """
    # 0) get the rotation matrix from optical flow
    if rotation_type == "2D":
        # compute the average of optical flow & get the delta theta and phi
        theta_delta, phi_delta = flow2rotation_2d(erp_flow, False)

        if not forward_warp:
            theta_delta = -theta_delta
            phi_delta = -phi_delta
        rotation_mat = spherical_coordinates.rot_sph2mat(theta_delta, phi_delta, False)
    elif rotation_type == "3D":
        rotation_mat = flow2rotation_3d(erp_flow)
        if not forward_warp:
            rotation_mat = rotation_mat.T
    else:
        log.error("Do not suport rotation type {}".format(rotation_type))

    # 1) rotate the ERP image with the rotation matrix
    erp_image_rot = sc.rotate_erp_array(erp_image, rotation_mat=rotation_mat)
    if erp_image.dtype == np.uint8:
        erp_image_rot = erp_image_rot.astype(np.uint8)

    return erp_image_rot, rotation_mat
