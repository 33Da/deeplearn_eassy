import os
import struct

from .logger import logger
from .crc32c import *


def directory_check(log_dir):
    """Initialize the directory for log files.
    
    :param log_dir: The directory to store log files.
    :type log_dir: str
    """
    if os.path.exists(log_dir):
        if os.path.isfile(log_dir):
            logger.error("{} is already exists, however, \
                it is not a directory.".format(log_dir))
            exit()
    else: 
        os.makedirs(log_dir)
     
     
class RecordWriter(object):
    def __init__(self, file_path):
        self.file_path= file_path
        self._writer = open(self.file_path, 'wb')
     
    def write(self, data):
        w = self._writer.write
        header = struct.pack('Q', len(data))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(data)
        w(struct.pack('I', masked_crc32c(data)))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

