import numpy as np
from scipy import ndimage

import gnomonic_projection
import polygon
import spherical_coordinates
import projection

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Cubemap for rgb image and optical flow:
1) 6 face order is +x, -x, +y, -y, +z, -z;   
Reference: https://en.wikipedia.org/wiki/Cube_mapping
"""


def get_cubemap_parameters(padding_size=0.0):
    """Get the information of circumscribed cuboid in spherical coordinate system:
    0) tangent points;
    1) 4 corner points for each tangent images;
    2) tangent area range in spherical coordinate. And the points order is: TL->TR->BR->BL.

    :param padding_size: the padding size is base on the tangent image scale. 
    :type: float
    :return: the faces parameters
    :rtype: dict
    """
    cubemap_point_phi = np.arctan(np.sqrt(2.0) * 0.5)  # the poler of the point

    # 1) get the tangent points (theta, phi)
    tangent_center_points_list = np.zeros((6, 2), dtype=float)
    tangent_center_points_list[0] = [np.pi / 2.0, 0.0]  # +x
    tangent_center_points_list[1] = [-np.pi / 2.0, 0.0]  # -x
    tangent_center_points_list[2] = [0.0, -np.pi / 2.0]  # +y
    tangent_center_points_list[3] = [0.0, np.pi / 2.0]  # -y
    tangent_center_points_list[4] = [0.0, 0.0]  # +z
    tangent_center_points_list[5] = [-np.pi, 0.0]  # -z

    # 2) circumscribed cuboidfor 6 face's 4 3D point (theta, phi), unite sphere
    face_points_sph_list = np.zeros((6, 4, 2), dtype=float)
    # Face 0, +x
    face_idx = 0
    face_points_sph_list[face_idx][0] = [0.25 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [0.25 * np.pi, -cubemap_point_phi]  # BL

    # Face 1, -x
    face_idx = 1
    face_points_sph_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.25 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [-0.25 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_phi]  # BL

    # Face 2, +y
    face_idx = 2
    face_points_sph_list[face_idx][0] = [-0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.25 * np.pi, cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.25 * np.pi, cubemap_point_phi]  # BL

    # Face 3, -y
    face_idx = 3
    face_points_sph_list[face_idx][0] = [-0.25 * np.pi, -cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.25 * np.pi, -cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.75 * np.pi, -cubemap_point_phi]  # BL

    # Face 4, +z
    face_idx = 4
    face_points_sph_list[face_idx][0] = [-0.25 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [0.25 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [0.25 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [-0.25 * np.pi, -cubemap_point_phi]  # BL

    # Face 5, -z
    face_idx = 5
    face_points_sph_list[face_idx][0] = [0.75 * np.pi, cubemap_point_phi]  # TL
    face_points_sph_list[face_idx][1] = [-0.75 * np.pi, cubemap_point_phi]  # TR
    face_points_sph_list[face_idx][2] = [-0.75 * np.pi, -cubemap_point_phi]  # BR
    face_points_sph_list[face_idx][3] = [0.75 * np.pi, -cubemap_point_phi]  # BL

    # 3) the cubemap face range in the ERP image, the tangent range order is -x, +x , -y, +y
    face_erp_range = np.zeros((6, 4, 2), dtype=float)
    tangent_padded_range = 1.0 + padding_size
    for index in range(0, len(tangent_center_points_list)):
        tangent_center_point = tangent_center_points_list[index]
        theta_0 = tangent_center_point[0]
        phi_0 = tangent_center_point[1]
        if index == 0 or index == 1 or index == 4 or index == 5:
            # +x, -x , +z, -z
            # theta range
            theta_min, _ = gnomonic_projection.reverse_gnomonic_projection(-tangent_padded_range, 0, theta_0, phi_0)
            theta_max, _ = gnomonic_projection.reverse_gnomonic_projection(tangent_padded_range, 0, theta_0, phi_0)
            # phi (latitude) range
            _, phi_max = gnomonic_projection.reverse_gnomonic_projection(0, tangent_padded_range, theta_0, phi_0)
            _, phi_min = gnomonic_projection.reverse_gnomonic_projection(0, -tangent_padded_range, theta_0, phi_0)
        elif index == 2 or index == 3:
            # +y, -y
            _, phi_ = gnomonic_projection.reverse_gnomonic_projection(-tangent_padded_range, tangent_padded_range, theta_0, phi_0)
            if index == 2:  # +y
                theta_min = -np.pi
                theta_max = np.pi
                phi_max = phi_
                phi_min = -np.pi/2
            if index == 3:  # -y
                theta_min = -np.pi
                theta_max = np.pi
                phi_max = np.pi/2
                phi_min = phi_

        # the each face range in spherical coordinate
        face_erp_range[index][0] = [theta_min, phi_max]  # TL
        face_erp_range[index][1] = [theta_max, phi_max]  # TR
        face_erp_range[index][2] = [theta_max, phi_min]  # BR
        face_erp_range[index][3] = [theta_min, phi_min]  # BL

    return {"tangent_points": tangent_center_points_list,
            "face_points": face_points_sph_list,
            "face_erp_range": face_erp_range}


def erp2cubemap_image(erp_image_mat, padding_size=0.0, face_image_size=None):
    """Project the equirectangular optical flow to 6 face of cube map base on the inverse gnomonic projection.
    The (x,y) of tangent image's tangent point is (0,0) of tangent image.

    :param erp_image_mat: the equirectangular image, dimension is [height, width, 3]
    :type erp_image_mat: numpy 
    :param  face_image_size: the tangent face image size 
    :type face_image_size: int
    :param padding_size: the padding size outside the face boundary, defaults to 0.0, do not padding
    :type padding_size: the bound, optional
    :retrun: 6 images of each fact of cubemap projection
    :rtype: list
    """
    # get the cube map with inverse of gnomonic projection.
    cubmap_tangent_images = []

    erp_image_height = np.shape(erp_image_mat)[0]
    erp_image_width = np.shape(erp_image_mat)[1]
    erp_image_channel = np.shape(erp_image_mat)[2]

    if face_image_size is None:
        face_image_size = int(erp_image_width / 4.0)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    pbc = 1.0 + padding_size  # projection_boundary_coefficient

    for index in range(0, 6):
        center_point = tangent_points_list[index]

        # tangent center project point
        theta_0 = center_point[0]
        phi_0 = center_point[1]

        # the xy of tangent image
        x_grid = np.linspace(-pbc, pbc, face_image_size)
        y_grid = np.linspace(pbc, -pbc, face_image_size)
        x, y = np.meshgrid(x_grid, y_grid)

        # get the value of pixel in the tangent image and the spherical coordinate location coresponding the tangent image (x,y)
        theta_, phi_ = gnomonic_projection.reverse_gnomonic_projection(x, y, theta_0, phi_0)

        # spherical coordinate to pixel location
        erp_pixel_x = ((theta_ + np.pi) / (2 * np.pi)) * erp_image_width
        erp_pixel_y = (-phi_ + np.pi / 2.0) / np.pi * erp_image_height

        # process warp around
        erp_pixel_x[erp_pixel_x < 0] = erp_pixel_x[erp_pixel_x < 0] + erp_image_width
        erp_pixel_x[erp_pixel_x >= erp_image_width] = erp_pixel_x[erp_pixel_x >= erp_image_width] - erp_image_width

        erp_pixel_y[erp_pixel_y < 0] = erp_pixel_y[erp_pixel_y < 0] + erp_image_height
        erp_pixel_y[erp_pixel_y >= erp_image_height] = erp_pixel_y[erp_pixel_y >= erp_image_height] - erp_image_height

        # interpollation
        face_image = np.zeros((face_image_size, face_image_size, erp_image_channel), dtype=float)
        for channel in range(0, erp_image_channel):
            face_image[:, :, channel] = ndimage.map_coordinates(erp_image_mat[:, :, channel], [erp_pixel_y, erp_pixel_x], order=1, mode='wrap')

        cubmap_tangent_images.append(face_image)

    return cubmap_tangent_images


def face_meshgrid(face_erp_range_sphere_list, erp_image_height):
    """The points in the face available area.

    :param face_erp_range_sphere_list: []
    :type face_erp_range_sphere_list: list
    :param erp_image_height: [description]
    :type erp_image_height: [type]
    :return: the available points list.
    :rtype: tuple
    """
    face_erp_range_sphere_array = np.array(face_erp_range_sphere_list)
    face_theta_min = face_erp_range_sphere_array[:, 0].min()
    face_theta_max = face_erp_range_sphere_array[:, 0].max()
    face_phi_min = face_erp_range_sphere_array[:, 1].min()
    face_phi_max = face_erp_range_sphere_array[:, 1].max()
    face_erp_x_min, face_erp_y_max = spherical_coordinates.sph2erp(face_theta_min, face_phi_min, erp_image_height, False)
    face_erp_x_max, face_erp_y_min = spherical_coordinates.sph2erp(face_theta_max, face_phi_max, erp_image_height, False)

    # process the image boundary
    if face_theta_min == -np.pi:
        face_erp_x_min = 0
    elif int(face_erp_x_min) > 0:
        face_erp_x_min = int(face_erp_x_min)
    else:
        face_erp_x_min = int(face_erp_x_min - 0.5)
    #
    if face_theta_max == np.pi:
        face_erp_x_max = erp_image_height * 2 - 1
    elif int(face_erp_x_max) > 0:
        face_erp_x_max = int(face_erp_x_max + 0.5)
    else:
        face_erp_x_max = int(face_erp_x_max)
    #
    if face_phi_max == np.pi * 0.5:
        face_erp_y_min = 0
    elif int(face_erp_y_min) >= 0:
        face_erp_y_min = int(face_erp_y_min)
    else:
        face_erp_y_min = int(face_erp_y_min - 0.5)
    #
    if face_phi_min == -np.pi * 0.5:
        face_erp_y_max = erp_image_height - 1
    if int(face_erp_y_max) > 0:
        face_erp_y_max = int(face_erp_y_max + 0.5)
    else:
        face_erp_y_max = int(face_erp_y_max)

    # 1) get ERP image's pix
    face_erp_x_grid = np.linspace(face_erp_x_min, face_erp_x_max, face_erp_x_max - face_erp_x_min + 1)
    face_erp_y_grid = np.linspace(face_erp_y_min, face_erp_y_max, face_erp_y_max - face_erp_y_min + 1)
    face_erp_x, face_erp_y = np.meshgrid(face_erp_x_grid, face_erp_y_grid)
    face_erp_x = np.remainder(face_erp_x, erp_image_height * 2)
    face_erp_y = np.remainder(face_erp_y, erp_image_height)
    return face_erp_x, face_erp_y


def cubemap2erp_flow(cubemap_flows_list, face_flows_wraparound=None, erp_image_height=None, padding_size=0.0, image_erp_src=None, image_erp_tar=None, wrap_around=False):
    """
    Assamble the 6 cubemap optical flow to ERP optical flow. 

    :param cubemap_flows_list: the images sequence is +x, -x, +y, -y, +z, -z
    :type cubemap_flows_list: list
    :param face_flows_wraparound: the optical flow wrap around index, List of Numpy, True need to wrap around, False is not.
    :type face_flows_wraparound: list
    :param erp_flow_height: the height of output flow 
    :type erp_flow_height: int
    :param padding_size: the cubemap's padding area size, defaults to 0.0
    :type padding_size: float, optional
    :param wrap_around: True, the optical flow is as perspective optical flow, False, it's warp around.
    :type wrap_around: bool
    :return: the ERP flow image the image size 
    :rtype: numpy
    """
    # check the face images number
    if not 6 == len(cubemap_flows_list):
        log.error("the cubemap images number is not 6")

    # get ERP image size
    cubemap_image_size = np.shape(cubemap_flows_list[0])[0]
    if erp_image_height is None:
        erp_image_height = cubemap_image_size * 2.0
    erp_image_height = int(erp_image_height)
    erp_flow_width = int(erp_image_height * 2.0)
    erp_flow_channel = np.shape(cubemap_flows_list[0])[2]
    if not erp_flow_channel == 2:
        log.error("The flow channels number is {}".format(erp_flow_channel))

    erp_flow_mat = np.zeros((erp_image_height, erp_flow_width, 2), dtype=np.float64)
    erp_flow_weight_mat = np.zeros((erp_image_height, erp_flow_width), dtype=np.float64)

    cubemap_points = get_cubemap_parameters(padding_size)
    tangent_points_list = cubemap_points["tangent_points"]
    face_erp_range_sphere_list = cubemap_points["face_erp_range"]
    pbc = 1.0 + padding_size  # projection_boundary_coefficient
    # gnomonic2image_ratio = (cubemap_image_size - 1) / (2.0 + padding_size * 2.0)

    for face_index in range(0, 6):
        # # 2) get the pixels location in tangent image location
        # get the tangent ERP image pixel's spherical coordinate location range for each face
        # ERP image space --> spherical space
        face_erp_x, face_erp_y = face_meshgrid(face_erp_range_sphere_list[face_index], erp_image_height)
        face_theta_, face_phi_ = spherical_coordinates.erp2sph((face_erp_x, face_erp_y), erp_image_height, False)

        # spherical space --> normailzed tangent image space
        theta_0, phi_0 = tangent_points_list[face_index]
        face_x_src_gnomonic, face_y_src_gnomonic, _ = gnomonic_projection.gnomonic_projection(face_theta_, face_phi_, theta_0, phi_0)

        # get ERP image's pixel available array, indicate pixels whether fall in the tangent face image, remove the pixels outside the tangent image
        available_list = polygon.inside_polygon_2d(np.stack((face_x_src_gnomonic.flatten(), face_y_src_gnomonic.flatten()), axis=1), np.array([
            [-pbc, pbc], [pbc, pbc], [pbc, -pbc], [-pbc, -pbc]]), True).reshape(np.shape(face_x_src_gnomonic))

        # normailzed tangent image space --> tangent image space
        tangent_gnomonic_range = [-pbc, +pbc, -pbc, +pbc]
        face_x_src_available, face_y_src_available = gnomonic_projection.gnomonic2pixel(
            face_x_src_gnomonic[available_list], face_y_src_gnomonic[available_list], 0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # 3) get the value of interpollations
        # 3-0) get the tangent images flow in the tangent image space
        face_flow_x = ndimage.map_coordinates(cubemap_flows_list[face_index][:, :, 0], [face_y_src_available, face_x_src_available], order=1, mode='constant', cval=0.0)
        face_x_tar_pixel_available = face_x_src_available + face_flow_x
        face_flow_y = ndimage.map_coordinates(cubemap_flows_list[face_index][:, :, 1], [face_y_src_available, face_x_src_available], order=1, mode='constant', cval=0.0)
        face_y_tar_pixel_available = face_y_src_available + face_flow_y

        face_tar_points_wraparound = None
        if face_flows_wraparound is not None:
            face_tar_points_wraparound = ndimage.map_coordinates(face_flows_wraparound[face_index], [face_y_src_available, face_x_src_available], order=1, mode='constant', cval=0.0)
            face_tar_points_wraparound = ~face_tar_points_wraparound

        # 3-1) transfrom the flow from tangent image space to ERP image space
        # tangent image space --> tangent normalized space
        face_x_tar_gnomonic_available, face_y_tar_gnomonic_available = gnomonic_projection.pixel2gnomonic(
            face_x_tar_pixel_available, face_y_tar_pixel_available, 0.0, cubemap_image_size, cubemap_image_size, tangent_gnomonic_range)

        # tangent normailzed space --> spherical space
        face_theta_tar, face_phi_tar = gnomonic_projection.reverse_gnomonic_projection(face_x_tar_gnomonic_available, face_y_tar_gnomonic_available, theta_0, phi_0, face_tar_points_wraparound)

        # spherical space --> ERP image space
        face_x_tar_available, face_y_tar_available = spherical_coordinates.sph2erp(face_theta_tar, face_phi_tar, erp_image_height, True)

        # 4) get ERP flow with source and target pixels location
        # 4-0) the ERP flow
        face_flow_u = face_x_tar_available - face_erp_x[available_list]
        face_flow_v = face_y_tar_available - face_erp_y[available_list]

        # 4-1) blend the optical flow
        # comput the all available pixels' weight
        if image_erp_src is not None and image_erp_tar is not None:
            log.debug("cubemap flow blender with weight.")
            weight_type = "image_warp_error"
            face_weight_mat_2 = projection.get_blend_weight_cubemap(face_erp_x[available_list], face_erp_y[available_list], weight_type,
                                                                    np.stack((face_x_tar_available, face_y_tar_available), axis=1), image_erp_src, image_erp_tar)
            face_weight_mat = face_weight_mat_2
        else:
            weight_type = "straightforward"
            face_weight_mat = projection.get_blend_weight_cubemap(face_x_src_gnomonic[available_list].flatten(), face_y_src_gnomonic[available_list].flatten(), weight_type)

        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 0] += face_flow_u * face_weight_mat
        erp_flow_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64), 1] += face_flow_v * face_weight_mat
        erp_flow_weight_mat[face_erp_y[available_list].astype(np.int64), face_erp_x[available_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    non_zero_weight_list = erp_flow_weight_mat != 0
    if not np.all(non_zero_weight_list):
        log.warn("the optical flow weight matrix contain 0.")
    for channel_index in range(0, 2):
        erp_flow_mat[:, :, channel_index][non_zero_weight_list] = erp_flow_mat[:, :, channel_index][non_zero_weight_list] / erp_flow_weight_mat[non_zero_weight_list]

    return erp_flow_mat
