import os

import numpy as np
from PIL import Image


from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def image_read(image_file_path):
    """Load rgb image data from file.

    :param image_file_path: the absolute path of image
    :type image_file_path: str
    :return: the numpy array of image
    :rtype: numpy
    """
    if not os.path.exists(image_file_path):
        log.error("{} do not exist.".format(image_file_path))

    return np.asarray(Image.open(image_file_path))


def image_save(image_data, image_file_path):
    """Save numpy array as image.

    :param image_data: Numpy array store image data. numpy 
    :type image_data: numpy
    :param image_file_path: The image's path
    :type image_file_path: str
    """
    # 0) convert the datatype
    image = None
    if image_data.dtype in [np.float, np.int64, np.int]:
        print("saved image array type is {}, converting to uint8".format(image_data.dtype))
        image = image_data.astype(np.uint8)
    else:
        image = image_data

    # 1) save to image file
    image_channels_number = image.shape[2]
    if image_channels_number == 3:
        im = Image.fromarray(image)
        im.save(image_file_path)
    else:
        log.error("The image channel number is {}".format(image_channels_number))
