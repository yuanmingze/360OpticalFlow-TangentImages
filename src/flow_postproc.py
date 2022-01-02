import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def erp_pixles_modulo(x_arrray, y_array, image_width, image_height):
    """ Make x,y and ERP pixels coordinate system range.
    """
    x_arrray_new = np.remainder(x_arrray + 0.5, image_width) - 0.5
    y_array_new = np.remainder(y_array + 0.5, image_height) - 0.5
    return x_arrray_new, y_array_new


def erp_of_wraparound(erp_flow, of_u_threshold=None):
    """Convert un-wrap-around (do not overflow) to ERP optical flow to the wrap-around (overflow) ERP optical flow.
    The optical flow larger than threshold need to be wrap-around.

    :param erp_flow: the panoramic optical flow [height, width, 2]
    :type  erp_flow: numpy
    :param of_u_threshold: The wrap-around threshold of optical flow u.
    :type of_u_threshold: float
    :return: The ERP optical flow
    :rtype: numpy
    """
    image_width = np.shape(erp_flow)[1]
    if of_u_threshold is None:
        of_u_threshold = image_width / 2.0

    flow_u = erp_flow[:, :, 0]
    # minus width
    index_minus_src_point_range = np.full(flow_u.shape[0:2], False)
    index_minus_src_point_range[:, 0: int(image_width - 1 - of_u_threshold)] = True
    index_minus_src_point = np.logical_and(index_minus_src_point_range, flow_u > of_u_threshold)
    flow_u[index_minus_src_point] = flow_u[index_minus_src_point] - image_width
    # plus width
    index_plus_src_point_range = np.full(flow_u.shape, False)
    index_plus_src_point_range[:, int(image_width - 1 - of_u_threshold): image_width - 1] = True
    index_plus_src_point = np.logical_and(index_plus_src_point_range, flow_u < -of_u_threshold)
    flow_u[index_plus_src_point] = flow_u[index_plus_src_point] + image_width

    return np.stack((flow_u, erp_flow[:, :, 1]), axis=2)
