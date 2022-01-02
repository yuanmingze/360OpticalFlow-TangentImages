import flow_postproc
import flow_vis
import flow_warp
import image_io
import projection
import projection_cubemap as proj_cm
import projection_icosahedron as proj_ico
import spherical_coordinates

import cv2
import numpy as np

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def of_methdod_DIS(image_src_original, image_tar_original):
    """Compute the DIS flow.

    :param image_src_original: The optical flow source image.
    :type image_src_original: numpy
    :param image_tar_original: The optical flow target image.
    :type image_tar_original: numpy
    :return: the optical flow.
    :rtype: numpy
    """
    image_src = image_src_original
    image_tar = image_tar_original

    # the image including alpha channel
    if image_src.shape[2] == 4:
        image_src = image_src[:, :, :3]
    if image_tar.shape[2] == 4:
        image_tar = image_tar[:, :, :3]

    # convert image dtype
    if image_src.dtype != np.uint8:
        image_src = image_src.astype(np.uint8)
    if image_tar.dtype != np.uint8:
        image_tar = image_tar.astype(np.uint8)

    # RGB to gray
    if image_src.shape[2] == 3:
        image_src_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_src_gray = image_src

    if image_tar.shape[2] == 3:
        image_tar_gray = cv2.cvtColor(image_tar, cv2.COLOR_BGR2GRAY)
        log.debug("the DIS input is gray, convert the RGB image to grapy.")
    else:
        image_tar_gray = image_tar

    inst = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(True)

    return inst.calc(image_src_gray, image_tar_gray, None)


def debug_save_of(of_data, output_filepath):
    """Visualize optical flow both warp-around and un-warp-around."""
    # flow_vis.flow_value_to_color(of_data)
    min_ratio = 0.2
    max_ratio = 0.8
    of_data_visual = flow_vis.flow_to_color(of_data, min_ratio=min_ratio, max_ratio=max_ratio)
    image_io.image_save(of_data_visual, output_filepath + "_flow_wraparound.jpg")
    # of_data_warparound = flow_postproc.erp_of_unwraparound(of_data)
    of_data_warparound = flow_postproc.erp_of_wraparound(of_data)
    of_data_warparound_visual = flow_vis.flow_to_color(of_data_warparound, min_ratio=min_ratio, max_ratio=max_ratio)
    image_io.image_save(of_data_warparound_visual, output_filepath + "_flow_unwraparound.jpg")


class PanoOpticalFlow():
    """
    The multi-steps method, transfer the global rotation warped ERP image to next step.
    The process use warp-around (overflowed) optical flow.

    Note: To prevent seam on the optical flow, process the wrap-around (overflowed) on the last step.
    """

    def __init__(self):

        # the optical flow estimation function.
        self.optical_flow_base_line_method = of_methdod_DIS

        # The gnomonic projection padding size of cubemap and icosahedron.
        self.padding_size_cubemap = 0.4
        self.padding_size_ico = 0.4

        self.erp_image_height = None

        # the tangent image size
        self.tangent_image_width_ico = 480

        self.flow2rotmat_method = "3D"

        self.debug_enable = False
        # The folder storing output debug information.
        self.debug_output_dir = None

        #  the weight to icosahedron blending faces "normwarp" or "straightforward", default is "straightforward".
        self.face_blending_method_ico = "normwarp"

        # each parts enable switch
        self.erp_enable = True
        self.cubemap_enable = True
        self.ico_enable = True

    def estimate(self, src_erp_image, tar_erp_image):
        """Compute the optical flow with multi-step and icosahedron projection.
        Without the 1st step ERP optical flow estimation.

        :param src_erp_image: the source ERP image data.
        :type src_erp_image: numpy
        :param tar_erp_image: the target ERP image data.
        :type tar_erp_image: numpy
        :return: the optical flow from source image to target image.
        :rtype: numpy
        """
        erp_image_height = src_erp_image.shape[0]

        erp_optical_flow = None
        rotation_mat_list = []
        # 0) compute flow in ERP image & wrap target image, make it align with the source ERP image.
        if self.erp_enable:
            log.debug("0) compute ERP image flow")
            # 1-1) erp optical flow
            optical_flow_erp = self.optical_flow_base_line_method(src_erp_image, tar_erp_image)
            # 1-2) warp target image
            tar_erp_image_rot_erp, rotation_mat_erp = flow_warp.global_rotation_warping(tar_erp_image, optical_flow_erp, forward_warp=False, rotation_type=self.flow2rotmat_method)
            log.debug("ERP optical flow rotation is {}".format(spherical_coordinates.rot_mat2sph(rotation_mat_erp)))

            # update the target image and store rotation matrix
            tar_erp_image = tar_erp_image_rot_erp
            rotation_mat_list.append(rotation_mat_erp)
            erp_optical_flow = optical_flow_erp

            if self.debug_output_dir is not None and self.debug_enable:
                debug_save_of(optical_flow_erp, self.debug_output_dir + "pano_of_0_of_erp")
                image_io.image_save(tar_erp_image_rot_erp, self.debug_output_dir + "pano_of_0_erp_rot.jpg")

        # 1) compute flow with cubemap projection & warp target image
        if self.cubemap_enable:
            log.debug("1) compute cubemap projection image flow")
            # 1-1) cube map optical flow
            cubeface_images_src_list = proj_cm.erp2cubemap_image(src_erp_image, self.padding_size_cubemap)
            cubeface_images_tar_list = proj_cm.erp2cubemap_image(tar_erp_image, self.padding_size_cubemap)
            cubemap_face_of_list = []
            for index in range(0, len(cubeface_images_src_list)):
                optical_flow_cubemap = self.optical_flow_base_line_method(cubeface_images_src_list[index], cubeface_images_tar_list[index])
                cubemap_face_of_list.append(optical_flow_cubemap)

            optical_flow_cubemap = proj_cm.cubemap2erp_flow(cubemap_face_of_list, erp_image_height=erp_image_height,
                                                            padding_size=self.padding_size_cubemap, image_erp_src=src_erp_image, image_erp_tar=tar_erp_image, wrap_around=True)

            # 1-2) warp target image
            tar_erp_image_rot_cubemap, rotation_mat_cubemap = flow_warp.global_rotation_warping(tar_erp_image, optical_flow_cubemap, forward_warp=False, rotation_type=self.flow2rotmat_method)
            log.debug("Cubemap optical flow rotation is {}".format(spherical_coordinates.rot_mat2sph(rotation_mat_cubemap)))

            # update the target image
            tar_erp_image = tar_erp_image_rot_cubemap
            rotation_mat_list.append(rotation_mat_cubemap)
            erp_optical_flow = optical_flow_cubemap

            # for debug
            if self.debug_output_dir is not None and self.debug_enable:
                debug_save_of(optical_flow_cubemap, self.debug_output_dir + "pano_of_0_of_cubemap")
                image_io.image_save(tar_erp_image_rot_cubemap, self.debug_output_dir + "pano_of_0_cubemap_rot.jpg")

        # 2) compute flow with icosahedron projection
        if self.ico_enable:
            log.debug("2) compute icosahedron projection image flow")
            # ico optical flow
            icoface_images_src_list = proj_ico.erp2ico_image(src_erp_image, self.tangent_image_width_ico, self.padding_size_ico, full_face_image=True)
            icoface_images_tar_list = proj_ico.erp2ico_image(tar_erp_image, self.tangent_image_width_ico, self.padding_size_ico, full_face_image=True)
            ico_face_of_list = []
            for index in range(0, len(icoface_images_src_list)):
                optical_flow_ico = self.optical_flow_base_line_method(icoface_images_src_list[index], icoface_images_tar_list[index])
                ico_face_of_list.append(optical_flow_ico)
            optical_flow_ico = proj_ico.ico2erp_flow(tangent_flows_list=ico_face_of_list,
                                                     erp_flow_height=erp_image_height,
                                                     padding_size=self.padding_size_ico,
                                                     image_erp_src=src_erp_image, image_erp_tar=tar_erp_image, wrap_around=True, face_blending_method=self.face_blending_method_ico)

            # update
            erp_optical_flow = optical_flow_ico
            if self.debug_output_dir is not None and self.debug_enable:
                # 2-2) warp target image
                debug_save_of(optical_flow_ico, self.debug_output_dir + "pano_of_0_of_ico")

        # 3) accumulate all-steps optical flow
        for rotation_mat in reversed(rotation_mat_list):
            erp_optical_flow = projection.flow_rotate_endpoint(erp_optical_flow, rotation_mat.T)

        return erp_optical_flow
