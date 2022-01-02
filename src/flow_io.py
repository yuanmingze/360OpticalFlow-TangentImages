import os
from struct import pack, unpack

import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def flow_write(flow_data, file_path):
    """Save optical flow data to file.

    :param flow_data: the optical flow data array, required size is (height, width, 2)
    :type flow_data: numpy
    :param file_path: the output file path.
    :type file_path: str
    """
    if os.path.exists(file_path):
        log.warn("file {} exist.".format(file_path))

    # get the file format from the extension name
    _, format_str = os.path.splitext(file_path)
    if format_str == ".flo":
        return write_flow_flo(flow_data, file_path)


def read_flow_flo(file_name):
    """Load optical flow from *.flo file.

    :param file_name: the flo file path.
    :type file_name: str
    :return: numpy array of shape (height, width, 2)
    :rtype: numpy
    """
    if not os.path.exists(file_name):
        log.error("{} do not exist!".format(file_name))

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(file_name)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in file_name %s' % file_name)
    assert ext == '.flo', exit('readFlowFile: file_name %s should have extension ''.flo''' % file_name)

    try:
        fid = open(file_name, 'rb')
    except IOError:
        log.error('readFlowFile: could not open %s', file_name)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % file_name)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (file_name, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (file_name, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow


def write_flow_flo(flow_data, fname):
    """Write the optical flow data to *.flo file.

    :param img: Optical flow data, [height, width, 2]
    :type img: numpy
    :param fname: the output file path.
    :type fname: str
    """
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = flow_data.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

    fid = None
    try:
        fid = open(fname, 'wb')
    except IOError:
        print('writeFlowFile: could not open %s', fname)

    # write the header
    fid.write(bytes(TAG_STRING, 'utf-8'))
    fid.write(pack('i', width))
    fid.write(pack('i', height))

    # arrange into matrix form
    tmp = np.zeros((height, width*nBands), np.float32)

    tmp[:, np.arange(width) * nBands] = flow_data[:, :, 0]
    tmp[:, np.arange(width) * nBands + 1] = np.squeeze(flow_data[:, :, 1])

    fid.write(bytes(tmp))

    fid.close()
