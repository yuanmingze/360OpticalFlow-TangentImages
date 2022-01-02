import numpy as np
from scipy import ndimage


import flow_postproc
import polygon
import spherical_coordinates as sc


from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def get_blend_weight_ico(face_x_src_gnomonic,
                         face_y_src_gnomonic,
                         weight_type,
                         flow_uv=None,
                         image_erp_src=None,
                         image_erp_tar=None,
                         gnomonic_bounding_box=None):
    """Compute the faces's weight.

    :param face_x_src_gnomonic: the pixel's gnomonic coordinate x in tangent image
    :type face_x_src_gnomonic: numpy, [pixel_number]
    :param face_y_src_gnomonic: the pixel's gnomonic coordinate y in tangent image
    :type face_y_src_gnomonic: numpy, [pixel_number]
    :param weight_type: The weight compute method, [straightforward|cartesian_distance]
    :type weight_type: str
    :param flow_uv: the tangent face forward optical flow which is in image coordinate, unit is pixel. [:, 2]
    :type flow_uv: numpy 
    :param image_erp_src: the source ERP rgb image, used to compute the optical flow warp error
    :type image_erp_src: numpy
    :param image_erp_tar: the target ERP rgb image used to compute the optical flow warp error
    :type image_erp_tar: numpy
    :param gnomonic_bounding_box: the available pixels area's bounding box
    :type gnomonic_bounding_box: list
    :return: the cubemap's face weight used to blend different faces to ERP image.
    :rtype: numpy
    """
    weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)

    if weight_type == "image_warp_error":
        # compute the weight base on the ERP RGB image warp match.
        # flow_uv: target image's pixels coordinate corresponding the warpped pixels
        pixels_number = face_x_src_gnomonic.shape[0]
        channel_number = image_erp_src.shape[2]
        image_erp_tar_image_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_src_image_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_warp_diff = np.zeros((pixels_number, channel_number), np.float)
        for channel in range(0, channel_number):
            image_erp_tar_image_flow[:, channel] = ndimage.map_coordinates(image_erp_tar[:, :, channel], [flow_uv[:, 1], flow_uv[:, 0]], order=1, mode='constant', cval=255)
            image_erp_src_image_flow[:, channel] = ndimage.map_coordinates(image_erp_src[:, :, channel], [face_y_src_gnomonic, face_x_src_gnomonic], order=1, mode='constant', cval=255)
            image_erp_warp_diff[:, channel] = np.absolute(image_erp_tar_image_flow[:, channel] - image_erp_src_image_flow[:, channel])

        image_erp_warp_diff = np.mean(image_erp_warp_diff, axis=1) / np.mean(image_erp_warp_diff)  # 255.0
        weight_map = np.exp(-image_erp_warp_diff)

    else:
        log.error("the weight method {} do not exist.".format(weight_type))
    return weight_map


def get_blend_weight_cubemap(face_x_src_gnomonic, face_y_src_gnomonic,
                             weight_type,
                             flow_uv=None,
                             image_erp_src=None,
                             image_erp_tar=None,
                             gnomonic_bounding_box=None):
    """Compute the faces's weight.

    :param face_x_src_gnomonic: the pixel's gnomonic coordinate x in tangent image
    :type face_x_src_gnomonic: numpy, [pixel_number]
    :param face_y_src_gnomonic: the pixel's gnomonic coordinate y in tangent image
    :type face_y_src_gnomonic: numpy, [pixel_number]
    :param weight_type: The weight compute method, [straightforward|cartesian_distance]
    :type weight_type: str
    :param flow_uv: the tangent face forward optical flow which is in image coordinate, unit is pixel.
    :type flow_uv: numpy 
    :param image_erp_src: the source ERP rgb image, used to compute the optical flow warp error
    :type: numpy
    :param image_erp_tar: the target ERP rgb image used to compute the optical flow warp error
    :type: numpy
    :param gnomonic_bounding_box: the available pixels area's bounding box
    :type: list
    :return: the cubemap's face weight used to blend different faces to ERP image.
    :rtype: numpy
    """
    weight_map = np.zeros(face_x_src_gnomonic.shape[0], dtype=np.float)

    if weight_type == "straightforward":
        # just set the pixels in this cube map face range is available. [-1, +1, -1, +1]
        if gnomonic_bounding_box is None:
            pbc = 1
            gnomonic_bounding_box = np.array([[-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]])
        available_list = polygon.inside_polygon_2d(np.stack((face_x_src_gnomonic, face_y_src_gnomonic), axis=1), gnomonic_bounding_box, on_line=True, eps=1e-7)
        weight_map[available_list] = 1.0
    elif weight_type == "image_warp_error":
        # compute the weight base on the ERP RGB image warp match.
        # flow_uv: target image's pixels coordinate corresponding the warpped pixels
        pixels_number = face_x_src_gnomonic.shape[0]
        channel_number = image_erp_src.shape[2]
        image_erp_tar_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_src_flow = np.zeros((pixels_number, channel_number), np.float)
        image_erp_warp_diff = np.zeros((pixels_number, channel_number), np.float)
        for channel in range(0, channel_number):
            image_erp_tar_flow[:, channel] = ndimage.map_coordinates(image_erp_tar[:, :, channel], [flow_uv[:, 1], flow_uv[:, 0]], order=1, mode='constant', cval=255)
            image_erp_src_flow[:, channel] = ndimage.map_coordinates(image_erp_src[:, :, channel], [face_y_src_gnomonic, face_x_src_gnomonic], order=1, mode='constant', cval=255)
            image_erp_warp_diff[:, channel] = np.absolute(image_erp_tar_flow[:, channel] - image_erp_src_flow[:, channel])

        rgb_diff = np.linalg.norm(image_erp_warp_diff, axis=1)
        non_zeros_index = rgb_diff != 0.0
        weight_map = np.ones(face_x_src_gnomonic.shape[0], dtype=np.float)
        weight_map[non_zeros_index] = 0.95 / rgb_diff[non_zeros_index]
    else:
        log.error("the weight method {} do not exist.".format(weight_type))
    return weight_map


def flow_rotate_endpoint(optical_flow, rotation, wraparound=False):
    """ Add the rotation offset to the end points of optical flow.

    :param optical_flow: the original optical flow, [height, width, 2]
    :type optical_flow: numpy
    :param rotation: the rotation of spherical coordinate in radian, [theta, phi] or rotation matrix.
    :type rotation: tuple
    :return: the new optical flow
    :rtype: numpy 
    """
    flow_height = optical_flow.shape[0]
    flow_width = optical_flow.shape[1]
    end_points_array_x = np.linspace(0, flow_width, flow_width, endpoint=False)
    end_points_array_y = np.linspace(0, flow_height, flow_height, endpoint=False)
    src_points_array_xv, src_points_array_yv = np.meshgrid(end_points_array_x, end_points_array_y)

    # get end point location in ERP coordinate
    end_points_array_xv, end_points_array_yv = flow_postproc.erp_pixles_modulo(src_points_array_xv + optical_flow[:, :, 0], src_points_array_yv + optical_flow[:, :, 1], flow_width, flow_height)

    end_points_array = None
    if isinstance(rotation, (list, tuple)):
        rotation_mat = sc.rot_sph2mat(rotation[0], rotation[1])
        end_points_array = sc.rotation2erp_motion_vector((flow_height, flow_width), rotation_mat, wraparound=True)
    elif isinstance(rotation, np.ndarray):
        end_points_array = sc.rotation2erp_motion_vector((flow_height, flow_width), rotation_matrix=rotation, wraparound=True)
    else:
        log.error("Do not support rotation data type {}.".format(type(rotation)))

    rotation_flow_u = ndimage.map_coordinates(end_points_array[:, :, 0], [end_points_array_yv, end_points_array_xv], order=1, mode='wrap')
    rotation_flow_v = ndimage.map_coordinates(end_points_array[:, :, 1], [end_points_array_yv, end_points_array_xv], order=1, mode='wrap')

    end_points_array_xv, end_points_array_yv = flow_postproc.erp_pixles_modulo(end_points_array_xv + rotation_flow_u, end_points_array_yv + rotation_flow_v, flow_width, flow_height)

    # erp pixles location to flow
    end_points_array_xv -= src_points_array_xv
    end_points_array_yv -= src_points_array_yv
    flow_rotated = np.stack((end_points_array_xv, end_points_array_yv), axis=-1)
    if wraparound:
        flow_rotated = flow_postproc.erp_of_wraparound(flow_rotated)
    return flow_rotated
