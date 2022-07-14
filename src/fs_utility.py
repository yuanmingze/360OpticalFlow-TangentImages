
import pathlib

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def dir_make(directory):
    """
    check the existence of directory, if not mkdir
    :param directory: the directory path
    :type directory: str
    """
    # check
    if isinstance(directory, str):
        directory_path = pathlib.Path(directory)
    elif isinstance(directory, pathlib.Path):
        directory_path = directory
    else:
        log.warn("Directory is neither str nor pathlib.Path {}".format(directory))
        return
    # create folder
    if not directory_path.exists():
        directory_path.mkdir()
    else:
        log.info("Directory {} exist".format(directory))

