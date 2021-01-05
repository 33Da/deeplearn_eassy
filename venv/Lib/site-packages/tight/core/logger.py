# Copyright (c) 2017 lululemon athletica Canada inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def info(*args, **kwargs):
    """
    Log a message using the system logger.

    :param args:
    :param kwargs:
    :return: None
    """
    message = kwargs.pop('message')
    logger.info(message)


def error(*args, **kwargs):
    message = kwargs.pop('message')
    logger.error(message)
