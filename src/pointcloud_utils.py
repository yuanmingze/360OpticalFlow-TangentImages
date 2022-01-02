
import numpy as np
from scipy.spatial.transform import Rotation as R

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

def correpairs2rotation(src_points, tar_points):
    """ Get the 3D rotation form 3D points corresponding relationship.

    :param src_points: The source 3D points, the shape is [points_number, 3], xyz
    :type src_points: numpy
    :param tar_points: The target 3D points, the shape is [points_number, 3], xyz
    :type tar_points: numpy
    :return: The 3D rotation matrix
    :rtype: numpy 
    """
    # ICP get transformation
    H = None
    if np.shape(src_points)[0] > np.shape(tar_points)[0]:
        H = np.dot(src_points[:np.shape(tar_points)[0], :].T, tar_points)
    elif np.shape(src_points)[0] < np.shape(tar_points)[0]:
        H = np.dot(src_points.T, tar_points[:np.shape(src_points)[0], :])
    else:
        H = np.dot(src_points.T, tar_points)

    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = np.dot(Vt.T, U.T)    # rotation matrix

    return rotation_matrix
