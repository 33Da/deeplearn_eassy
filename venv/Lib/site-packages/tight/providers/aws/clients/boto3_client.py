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

import boto3
import sys
boto3_session = None


def session():
    current_module = sys.modules[__name__]
    boto3_session = getattr(current_module, 'boto3_session')
    if boto3_session:
        return boto3_session
    else:
        boto3_session = boto3.Session()
        setattr(current_module, 'boto3_session', boto3_session)
        return boto3_session
