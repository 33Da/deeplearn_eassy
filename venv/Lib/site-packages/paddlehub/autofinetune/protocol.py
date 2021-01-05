# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import threading
from enum import Enum
from .common import multi_thread_enabled


class CommandType(Enum):
    # in
    Initialize = b'IN'
    RequestTrialJobs = b'GE'
    ReportMetricData = b'ME'
    UpdateSearchSpace = b'SS'
    TrialEnd = b'EN'
    Terminate = b'TE'

    # out
    Initialized = b'ID'
    NewTrialJob = b'TR'
    SendTrialJobParameter = b'SP'
    NoMoreTrialJobs = b'NO'
    KillTrialJob = b'KI'

_lock = threading.Lock()
try:
    _in_file = open(3, 'rb')
    _out_file = open(4, 'wb')
except OSError:
    _msg = 'IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?'
    import logging
    logging.getLogger(__name__).warning(_msg)


def send(command, data):
    """
    Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        assert len(data) < 1000000, 'Command too long'
        msg = b'%b%06d%b' % (command.value, len(data), data)
        logging.getLogger(__name__).debug('Sending command, data: [%s]' % msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """
    Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    header = _in_file.read(8)
    logging.getLogger(__name__).debug('Received command, header: [%s]' % header)
    if header is None or len(header) < 8:
        # Pipe EOF encountered
        logging.getLogger(__name__).debug('Pipe EOF encountered')
        return None, None
    length = int(header[2:])
    data = _in_file.read(length)
    command = CommandType(header[:2])
    data = data.decode('utf8')
    logging.getLogger(__name__).debug('Received command, data: [%s]' % data)
    return command, data
