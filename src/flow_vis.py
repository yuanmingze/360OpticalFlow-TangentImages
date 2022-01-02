import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import image_evaluate
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


"""
 optical flow visualization;
reference 
https://github.com/tomrunia/OpticalFlow_Visualization
"""


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    :return: Color wheel
    :rtype: numpy
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def create_colorwheel_bar(image_size):
    """Make a color wheel."""
    x_list = np.linspace(-1.0, 1.0, num=image_size, endpoint=False)
    y_list = np.linspace(-1.0, 1.0, num=image_size, endpoint=False)
    xv, yv = np.meshgrid(x_list, y_list)
    flow_wheel = flow_uv_to_colors(xv, yv)
    return flow_wheel


def flow_uv_to_colors(u, v, convert_to_bgr=False, sph_of=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Spherical optical flow the u and v is pixel offset on ERP image space, do not need to normalize.

    :param u: If spherical, u is pixel offset. if perspectiveInput, it's normalized horizontal flow, shape [H,W].
    :type u: numpy
    :param v: If spherical, u is pixel offset. if perspectiveInput, it's normalized vertical flow, shape [H,W].
    :type v: numpy
    :param convert_to_bgr: Convert output image to BGR. Defaults to False.
    :type convert_to_bgr: bool, optional
    :param sph_of: if True use the spherical optical flow
    :type sph_of: 
    :return: Flow visualization image of shape [H,W,3]
    :rtype: numpy
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    row_number = colorwheel.shape[0]  # ncols is 55
    angle = None
    if sph_of:
        raise NotImplemented
    else:
        rad = np.sqrt(np.square(u) + np.square(v))
        angle = np.arctan2(-v, -u) / np.pi  # [-1, +1]
    angle_row_idx = (angle + 1) / 2 * (row_number - 1)
    angle_row_idx_floor = np.floor(angle_row_idx).astype(np.int32)
    angle_row_idx_ceil = angle_row_idx_floor + 1
    angle_row_idx_ceil[angle_row_idx_ceil == row_number] = 0
    ratio = angle_row_idx - angle_row_idx_floor
    for i in range(colorwheel.shape[1]):
        colorwheel_channel = colorwheel[:, i]
        row_floor = colorwheel_channel[angle_row_idx_floor] / 255.0
        row_ceiling = colorwheel_channel[angle_row_idx_ceil] / 255.0
        color = (1-ratio)*row_floor + ratio*row_ceiling  # bilinear interpolation
        idx = (rad <= 1)
        color[idx] = 1 - rad[idx] * (1-color[idx])
        color[~idx] = color[~idx] * 0.75   # radian larger than 1, out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * color)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False, min_ratio=0.0, max_ratio=1.0, add_bar=False, sph_of=False):
    """ Expects a two dimensional flow image of shape.

    :param flow_uv: Flow UV image of shape [H,W,2]
    :type flow_uv: numpy
    :param clip_flow: Clip maximum of flow values. Defaults to None e.g. [-100, 100]
    :type clip_flow: float
    :param convert_to_bgr: Convert output image to BGR. Defaults to False.
    :type convert_to_bgr: bool, optional
    :param min_ratio:
    :param sph_of: If Yes, it is visualize spherical optical flow.
    :return: Flow visualization image of shape [H,W,3]
    :rtype: numpy
    """
    # get the clip range
    if min_ratio != 0 and max_ratio != 1.0:
        clip_flow = image_evaluate.get_min_max(flow_uv, min_ratio, max_ratio)

    # visualize optical flow
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, clip_flow[0], clip_flow[1])
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    if sph_of:
        u_norm = u
        v_norm = v
    else:
        # normalize optical flow
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u_norm = u / (rad_max + epsilon)
        v_norm = v / (rad_max + epsilon)
    flow_colored = flow_uv_to_colors(u_norm, v_norm, convert_to_bgr, sph_of)
    # add color wheel and range to image
    if add_bar:
        fig, ax = plt.subplots()
        from matplotlib.offsetbox import (TextArea, OffsetImage, AnnotationBbox)
        # annotate the rad, u and v range
        raduv_range = "Rad Max: {}\nRad Min: {}\nu_min: {}\nu_max: {}\nv_min: {}\nv_max: {}".format(
            rad.max(), rad.min(), u.min(), u.max(), v.min(), v.max())
        annotation_text = TextArea(raduv_range)
        annotation_ab = AnnotationBbox(annotation_text,
                                       xycoords='axes fraction',
                                       pad=0.1,
                                       xy=(1.01, 0.8),
                                       box_alignment=(0., 0.5))
        ax.add_artist(annotation_ab)
        # add color wheel
        colorwheel = create_colorwheel_bar(150)
        imagebox = OffsetImage(colorwheel, zoom=1.0)
        imagebox.image.axes = ax
        colorwheel_ab = AnnotationBbox(imagebox,
                                       xycoords='axes fraction',
                                       pad=0.1,
                                       xy=(1.01, 0.3),
                                       box_alignment=(0., 0.5))
        ax.add_artist(colorwheel_ab)
        ax.imshow(flow_colored)
        fig.dpi = 100.0
        _, hi = fig.get_size_inches()
        fig.set_size_inches(hi * (10.0 / 4.0), hi)

        # convet to numpy
        fig.canvas.draw()  # draw the renderer
        w, h = fig.canvas.get_width_height()  # Get the RGBA buffer from the figure
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (w, h, 3)
        image = Image.frombytes("RGB", (w, h), buf.tostring())
        flow_colored = np.asarray(image)
        # plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close("all")
    return flow_colored
