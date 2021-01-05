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

from flywheel import Model, Field, Engine
import os
import sys
current_module = sys.modules[__name__]
session = None
engine = None


def connect(*args, **kwargs):
    engine = getattr(current_module, 'engine')
    session = getattr(current_module, 'session') or kwargs.pop('session', None)
    if engine:
        return engine
    else:
        engine = Engine()
        # Connect the engine to either a local DynamoDB or a particular region.
        if ('USE_LOCAL_DB' in os.environ and os.environ['USE_LOCAL_DB'] == 'True'):
            engine.connect(os.environ['AWS_REGION'], host='localhost',
                           port=8000,
                           access_key='anything',
                           secret_key='anything',
                           is_secure=False,
                           session=session)
        elif ('CI' in os.environ and os.environ['CI'] == 'True'):
            engine.connect_to_region(os.environ['AWS_REGION'], session=session)
        else:
            engine.connect_to_region(os.environ['AWS_REGION'])
        setattr(current_module, 'engine', engine)
        return engine
