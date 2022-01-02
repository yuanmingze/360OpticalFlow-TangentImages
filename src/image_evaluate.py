import numpy as np

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def get_min_max(data, min_ratio=0.0, max_ratio=1.0):
    """Get the max and min based on the ratio. 

    :param data: data array.
    :type data: numpy
    :param min_ratio: The ratio of minimum value, defaults to 0.0
    :type min_ratio: float, optional
    :param max_ratio: The ratio of maximum value, defaults to 1.0
    :type max_ratio: float, optional
    """
    vmin_ = 0
    vmax_ = 0
    if min_ratio != 0.0 or max_ratio != 1.0:
        flow_array = data.flatten()
        vmin_idx = int(flow_array.size * min_ratio)
        vmax_idx = int((flow_array.size - 1) * max_ratio)
        vmin_ = np.partition(flow_array, vmin_idx)[vmin_idx]
        vmax_ = np.partition(flow_array, vmax_idx)[vmax_idx]
        if min_ratio != 0 or max_ratio != 1.0:
            log.warn("clamp the optical flow value form [{},{}] to [{},{}]".format(np.amin(data), np.amax(data), vmin_, vmax_))
    else:
        vmin_ = np.amin(data)
        vmax_ = np.amax(data)

    return vmin_, vmax_
