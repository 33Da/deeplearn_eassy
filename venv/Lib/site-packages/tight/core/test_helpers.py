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

import pytest
import os
import importlib
import yaml
import json
from botocore import session as boto_session
from tight.providers.aws.clients import dynamo_db


@pytest.fixture
def app():
    return importlib.import_module('app_index')


@pytest.fixture
def event():
    with open('tests/fixtures/lambda_proxy_event.yml') as data_file:
        event = yaml.load(data_file)
    return event


def expected_response_body(dir, expectation_file, actual_response):
    file_path = '/'.join([dir, expectation_file])
    if 'PLAYBACK' in os.environ and os.environ['PLAYBACK'] == 'True':
        return json.loads(yaml.load(open(file_path))['body'])
    if 'RECORD' in os.environ and os.environ['RECORD'] == 'True':
        stream = file(file_path, 'w')
        yaml.safe_dump(actual_response, stream)
        return json.loads(actual_response['body'])


@pytest.fixture
def dynamo_db_session():
    session = getattr(dynamo_db, 'session') or None
    if session:
        return session
    else:
        session = boto_session.get_session()
        session.events = session.get_component('event_emitter')
        dynamo_db.session = session
        return session
