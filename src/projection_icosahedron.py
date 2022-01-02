import copy

import numpy as np
from scipy import ndimage
from skimage.transform import resize

import gnomonic_projection as gp
import spherical_coordinates as sc
import polygon
import projection
import flow_postproc

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def get_icosahedron_parameters(triangle_index, padding_size=0.0):
    """Get icosahedron's tangent face's paramters.
    Get the tangent point theta and phi. Known as the theta_0 and phi_0.
    The erp image origin as top-left corner

    :return the tangent face's tangent point and 3 vertices's location.
    """
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # the tangent point
    theta_0 = None
    phi_0 = None

    # the 3 points of tangent triangle in spherical coordinate
    triangle_point_00_theta = None
    triangle_point_00_phi = None
    triangle_point_01_theta = None
    triangle_point_01_phi = None
    triangle_point_02_theta = None
    triangle_point_02_phi = None

    theta_step = 2.0 * np.pi / 5.0
    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        # tangent point of inscribed spheric
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index * theta_step
        phi_0 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * theta_step
        triangle_point_01_phi = np.pi / 2.0
        triangle_point_02_theta = -np.pi + (triangle_index + 1) * theta_step
        triangle_point_02_phi = np.arctan(0.5)

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index_temp = triangle_index - 5
        # tangent point of inscribed spheric
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        phi_0 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + (triangle_index_temp + 1) * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = -np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.arctan(0.5)

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index_temp = triangle_index - 10
        # tangent point of inscribed spheric
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        if triangle_index_temp == 10:
            # cross the ERP image boundary
            triangle_point_00_theta = triangle_point_00_theta + 2 * np.pi
        triangle_point_01_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.arctan(0.5)

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index_temp = triangle_index - 15
        # tangent point of inscribed spheric
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_01_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        # cross the ERP image boundary
        if triangle_index_temp == 15:
            triangle_point_01_theta = triangle_point_01_theta + 2 * np.pi
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_02_theta = - np.pi + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.pi / 2.0

    tangent_point = [theta_0, phi_0]

    # the 3 points gnomonic coordinate in tangent image's gnomonic space
    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_theta, triangle_point_00_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_theta, triangle_point_01_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_theta, triangle_point_02_phi, theta_0, phi_0))

    # pading the tangent image
    triangle_points_tangent_pading = polygon.enlarge_polygon(triangle_points_tangent, padding_size)

    # if padding_size != 0.0:
    triangle_points_tangent = copy.deepcopy(triangle_points_tangent_pading)

    # the points in spherical location
    triangle_points_sph = []
    for index in range(3):
        tri_pading_x, tri_pading_y = triangle_points_tangent_pading[index]
        triangle_point_theta, triangle_point_phi = gp.reverse_gnomonic_projection(tri_pading_x, tri_pading_y, theta_0, phi_0)
        triangle_points_sph.append([triangle_point_theta, triangle_point_phi])

    # compute bounding box of the face in spherical coordinate
    availied_sph_area = []
    availied_sph_area = np.array(copy.deepcopy(triangle_points_sph))
    triangle_points_tangent_pading = np.array(triangle_points_tangent_pading)
    point_insert_x = np.sort(triangle_points_tangent_pading[:, 0])[1]
    point_insert_y = np.sort(triangle_points_tangent_pading[:, 1])[1]
    availied_sph_area = np.append(availied_sph_area, [gp.reverse_gnomonic_projection(point_insert_x, point_insert_y, theta_0, phi_0)], axis=0)
    # the bounding box of the face with spherical coordinate
    availied_ERP_area_sph = []  # [min_theta, max_theta, min_phi, max_phi]
    if 0 <= triangle_index <= 4:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.pi / 2.0)
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down
    elif 15 <= triangle_index <= 19:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(-np.pi / 2.0)
    else:
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))

    return {"tangent_point": tangent_point, "triangle_points_tangent": triangle_points_tangent, "triangle_points_sph": triangle_points_sph, "availied_ERP_area": availied_ERP_area_sph}


def erp2ico_image(erp_image, tangent_image_width, padding_size=0.0, full_face_image=True):
    """Project the equirectangular image to 20 triangle images.
    Project the equirectangular image to level-0 icosahedron.

    :param erp_image: the input equirectangular image.
    :type erp_image: numpy array, [height, width, 3]
    :param tangent_image_width: the output triangle image size, defaults to 480
    :type tangent_image_width: int, optional
    :param padding_size: the output face image' padding size
    :type padding_size: float
    :param full_face_image: If yes project all pixels in the face image, no just project the pixels in the face triangle, defaults to False
    :type full_face_image: bool, optional
    :return: a list contain 20 triangle images, the image is 4 channels, invalided pixel's alpha is 0, others is 1
    :type list
    """
    if len(erp_image.shape) == 3:
        if np.shape(erp_image)[2] == 4:
            erp_image = erp_image[:, :, 0:3]
    elif len(erp_image.shape) == 2:
        log.info("project single channel disp or depth map")
        erp_image = np.expand_dims(erp_image, axis=2)

    # ERP image size
    erp_image_height = np.shape(erp_image)[0]
    erp_image_width = np.shape(erp_image)[1]
    channel_number = np.shape(erp_image)[2]

    if erp_image_width != erp_image_height * 2:
        log.error("the ERP image dimession is {}".format(np.shape(erp_image)))

    tangent_image_list = []
    tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)

    # generate tangent images
    for triangle_index in range(0, 20):
        log.debug("generate the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        # the face gnomonic range in tangent space
        gnomonic_x_min = np.amin(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_x_max = np.amax(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_y_min = np.amin(tangent_triangle_vertices[:, 1], axis=0)
        gnomonic_y_max = np.amax(tangent_triangle_vertices[:, 1], axis=0)
        gnom_range_x = np.linspace(gnomonic_x_min, gnomonic_x_max, num=tangent_image_width, endpoint=True)
        gnom_range_y = np.linspace(gnomonic_y_max, gnomonic_y_min, num=tangent_image_height, endpoint=True)

        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        # the tangent triangle points coordinate in tangent image
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=np.bool)
        if not full_face_image:
            gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
            pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
            inside_list = polygon.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(gnom_range_xv.shape)

        # project to tangent image
        tangent_point = triangle_param["tangent_point"]
        tangent_triangle_theta_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_point[0], tangent_point[1])

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2erp(tangent_triangle_theta_, tangent_triangle_phi_, erp_image_height, sph_modulo=True)

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                             0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        if channel_number == 1:
            tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], 255)
        elif channel_number == 3:
            tangent_image = np.full([tangent_image_height, tangent_image_width, 4], 255)
        else:
            log.error("The channel number is {}".format(channel_number))

        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y, tangent_image_x, channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255)

        # set the pixels outside the boundary to transparent
        tangent_image[:, :, 3] = 0
        tangent_image[tangent_image_y, tangent_image_x, 3] = 255
        tangent_image_list.append(tangent_image)

    return tangent_image_list


def ico2erp_flow(tangent_flows_list, erp_flow_height=None, padding_size=0.0, image_erp_src=None, image_erp_tar=None, wrap_around=False, face_blending_method="straightforward"):
    """Stitch all 20 tangent flows to a ERP flow.

    :param tangent_flows_list: The list of 20 tangnet flow data.
    :type tangent_flows_list: list of numpy 
    :param erp_flow_height: the height of stitched ERP flow image 
    :type erp_flow_height: int
    :param padding_size: the each face's flow padding area size, defaults to 0.0
    :type padding_size: float, optional
    :return: the stitched ERP flow
    :rtype: numpy
    """
    # check the face images number
    if not 20 == len(tangent_flows_list):
        log.error("the ico face flow number is not 20")

    # get ERP and face's flow parameters
    tangent_flow_height = np.shape(tangent_flows_list[0])[0]
    tangent_flow_width = np.shape(tangent_flows_list[0])[1]
    if erp_flow_height is None:
        erp_flow_height = int(tangent_flow_height * 2.0)
    erp_flow_height = int(erp_flow_height)
    erp_flow_width = int(erp_flow_height * 2.0)
    erp_flow_channel = np.shape(tangent_flows_list[0])[2]
    if not erp_flow_channel == 2:
        log.error("The flow channels number is {}".format(erp_flow_channel))

    # erp flow and blending weight
    erp_flow_mat = np.zeros((erp_flow_height, erp_flow_width, 2), dtype=np.float64)
    erp_flow_weight_mat = np.zeros((erp_flow_height, erp_flow_width), dtype=np.float64)

    for face_index in range(0, len(tangent_flows_list)):
        # for triangle_index in range(0, 2):
        log.debug("stitch the tangent image {}".format(face_index))
        face_param = get_icosahedron_parameters(face_index, padding_size)
        theta_0 = face_param["tangent_point"][0]
        phi_0 = face_param["tangent_point"][1]
        triangle_points_tangent = np.array(face_param["triangle_points_tangent"])
        availied_ERP_area = face_param["availied_ERP_area"]

        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)
        face_src_range_gnom = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        pixel_eps = abs(gnomonic_x_max - gnomonic_x_min) / (2 * tangent_flow_width)

        # 1) get tangent face available pixles range in ERP spherical coordinate
        erp_flow_col_start, erp_flow_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_flow_height, sph_modulo=False)
        erp_flow_col_stop, erp_flow_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_flow_height, sph_modulo=False)
        # process the tangent flow boundary
        erp_flow_col_start = int(erp_flow_col_start) if int(erp_flow_col_start) > 0 else int(erp_flow_col_start - 0.5)
        erp_flow_col_stop = int(erp_flow_col_stop + 0.5) if int(erp_flow_col_stop) > 0 else int(erp_flow_col_stop)
        erp_flow_row_start = int(erp_flow_row_start) if int(erp_flow_row_start) > 0 else int(erp_flow_row_start - 0.5)
        erp_flow_row_stop = int(erp_flow_row_stop + 0.5) if int(erp_flow_row_stop) > 0 else int(erp_flow_row_stop)
        triangle_x_range = np.linspace(erp_flow_col_start, erp_flow_col_stop, erp_flow_col_stop - erp_flow_col_start + 1, endpoint=True)
        triangle_y_range = np.linspace(erp_flow_row_start, erp_flow_row_stop, erp_flow_row_stop - erp_flow_row_start + 1, endpoint=True)
        face_src_x_erp, face_src_y_erp = np.meshgrid(triangle_x_range, triangle_y_range)

        face_src_x_erp, face_src_y_erp = flow_postproc.erp_pixles_modulo(face_src_x_erp, face_src_y_erp, erp_flow_width, erp_flow_height)

        # 2) get the pixels location in tangent image location
        # ERP image space --> spherical space
        face_src_xy_sph = sc.erp2sph((face_src_x_erp, face_src_y_erp), erp_flow_height, sph_modulo=False)

        # spherical space --> normailzed tangent image space
        face_src_x_gnom, face_src_y_gnom, _ = gp.gnomonic_projection(face_src_xy_sph[0, :, :], face_src_xy_sph[1, :, :], theta_0, phi_0, True)

        # the available (in the triangle) pixels list

        available_list = np.full(face_src_x_gnom.shape, False)
        face_src_gnom_no_nan = np.logical_not(np.isnan(face_src_x_gnom))
        available_list[face_src_gnom_no_nan] = polygon.inside_polygon_2d(
            np.stack((face_src_x_gnom[face_src_gnom_no_nan].flatten(), face_src_y_gnom[face_src_gnom_no_nan].flatten()), axis=1),
            triangle_points_tangent, on_line=True, eps=pixel_eps)

        # normailzed tangent image space --> tangent image space
        face_src_x_gnom_pixel, face_src_y_gnom_pixel = gp.gnomonic2pixel(
            face_src_x_gnom[available_list], face_src_y_gnom[available_list], 0.0, tangent_flow_width, tangent_flow_height, face_src_range_gnom)

        # 3) get the value of optical flow end point location
        # 3-0) get the tangent images flow in the tangent image space, ignore the pixels outside the tangent image
        face_flow_u_gnom_pixel = ndimage.map_coordinates(tangent_flows_list[face_index][:, :, 0], [face_src_y_gnom_pixel,
                                                         face_src_x_gnom_pixel], order=1,  mode='nearest')  # mode='constant', cval=255)
        face_tar_x_gnom_pixel_avail = face_src_x_gnom_pixel + face_flow_u_gnom_pixel
        face_flow_v_gnom_pixel = ndimage.map_coordinates(tangent_flows_list[face_index][:, :, 1], [face_src_y_gnom_pixel, face_src_x_gnom_pixel], order=1, mode='nearest')  # mode='constant', cval=255)
        face_tar_y_gnom_pixel_avail = face_src_y_gnom_pixel + face_flow_v_gnom_pixel

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        face_tar_x_gnom_avail, face_tar_y_gnom_avail = gp.pixel2gnomonic(face_tar_x_gnom_pixel_avail, face_tar_y_gnom_pixel_avail, 0.0, tangent_flow_width, tangent_flow_height, face_src_range_gnom)
        # tangent normailzed space --> spherical space
        face_tar_x_sph_avail, face_tar_y_sph_avail = gp.reverse_gnomonic_projection(face_tar_x_gnom_avail, face_tar_y_gnom_avail, theta_0, phi_0)
        face_tar_x_sph_avail, face_tar_y_sph_avail = sc.sph_coord_modulo(face_tar_x_sph_avail, face_tar_y_sph_avail)

        # 3-2) process the optical flow wrap-around, including face, use the shorted path as real path.
        face_tar_x_erp, face_tar_y_erp = sc.sph2erp(face_tar_x_sph_avail, face_tar_y_sph_avail, erp_flow_height, sph_modulo=True)

        # # Process the face lines cross the ERP image boundary
        if wrap_around:
            face_src_x_sph_avail = face_src_xy_sph[0, :, :][available_list]
            face_src_y_sph_avail = face_src_xy_sph[1, :, :][available_list]
            face_src_x_sph_avail, face_src_y_sph_avail = sc.sph_coord_modulo(face_src_x_sph_avail, face_src_y_sph_avail)

            long_line = np.abs(face_tar_x_sph_avail - face_src_x_sph_avail) > np.pi
            cross_x_axis_minus2pi = np.logical_and(face_src_x_sph_avail < 0, face_tar_x_sph_avail > 0)
            cross_x_axis_minus2pi = np.logical_and(long_line, cross_x_axis_minus2pi)
            cross_x_axis_plus2pi = np.logical_and(face_src_x_sph_avail > 0, face_tar_x_sph_avail < 0)
            cross_x_axis_plus2pi = np.logical_and(long_line, cross_x_axis_plus2pi)
            face_tar_x_erp[cross_x_axis_minus2pi] = face_tar_x_erp[cross_x_axis_minus2pi] - erp_flow_width
            face_tar_x_erp[cross_x_axis_plus2pi] = face_tar_x_erp[cross_x_axis_plus2pi] + erp_flow_width

        # 4) get ERP flow with source and target pixels location
        # 4-0) the ERP flow
        face_flow_u_erp = face_tar_x_erp - face_src_x_erp[available_list]
        face_flow_v_erp = face_tar_y_erp - face_src_y_erp[available_list]

        # 4-1) compute the all available pixels' weight to blend the optical flow
        if face_blending_method == "straightforward":
            face_weight_mat = np.ones(face_src_y_gnom_pixel.shape, dtype=np.float64)

        elif face_blending_method == "normwarp":
            # resize the erp image
            if image_erp_src.shape[:2] != [erp_flow_height, erp_flow_width]:
                image_erp_src = resize(image_erp_src, (erp_flow_height, erp_flow_width), preserve_range=True)
            if image_erp_tar.shape[:2] != [erp_flow_height, erp_flow_width]:
                image_erp_tar = resize(image_erp_tar, (erp_flow_height, erp_flow_width), preserve_range=True)
            face_weight_mat_2 = projection.get_blend_weight_ico(face_src_x_erp[available_list], face_src_y_erp[available_list],
                                                                "image_warp_error", np.stack((face_tar_x_erp, face_tar_y_erp), axis=1),
                                                                image_erp_src, image_erp_tar)
            face_weight_mat = face_weight_mat_2

        # blender ERP flow and weight
        erp_flow_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64), 0] += face_flow_u_erp * face_weight_mat
        erp_flow_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64), 1] += face_flow_v_erp * face_weight_mat

        erp_flow_weight_mat[face_src_y_erp[available_list].astype(np.int64), face_src_x_erp[available_list].astype(np.int64)] += face_weight_mat

    non_zero_weight_list = erp_flow_weight_mat != 0.0
    if not np.all(non_zero_weight_list):
        log.warn("the optical flow weight matrix contain 0.")
    for channel_index in range(0, 2):
        erp_flow_mat[:, :, channel_index][non_zero_weight_list] = erp_flow_mat[:, :, channel_index][non_zero_weight_list] / erp_flow_weight_mat[non_zero_weight_list]

    if wrap_around:
        erp_flow_mat = flow_postproc.erp_of_wraparound(erp_flow_mat)

    return erp_flow_mat
