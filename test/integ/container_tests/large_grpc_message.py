#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import json

import requests


def test_grpc_message_4m_json():
    # this will generate a request just over the original
    # 4MB limit (1024 * 1024 * 1 * 4 bytes) + overhead
    # this matches the message size from the original issue report
    # response will have same size, so we are testing both ends
    data = {
        'shape': [1, 1024, 1024, 1],
        'dtype': 'float32'
    }

    response = requests.post("http://localhost:8080/invocations",
                             data=json.dumps(data),
                             headers={'Content-type': 'application/json',
                                      'Accept': 'application/json'}).content

    prediction = json.loads(response)

    expected_shape = {
        'dim': [
            {'size': '1'},
            {'size': '1024'},
            {'size': '1024'},
            {'size': '1'},
        ]
    }

    assert expected_shape == prediction['outputs']['y']['tensorShape']
    assert 2.0 == prediction['outputs']['y']['floatVal'][-1]


def test_large_grpc_message_512m_pb2():
    # this will generate request ~ 512mb
    # (1024 * 1024 * 128 * 4 bytes) + overhead
    # response will have same size, so we are testing both ends
    # returning bytes (serialized pb2) instead of json, because
    # our default json output function runs out of memory with
    # much smaller messages (around 128MB on if gunicorn running in 8GB)
    data = {
        'shape': [1, 1024, 1024, 128],
        'dtype': 'float32'
    }

    response = requests.post("http://localhost:8080/invocations",
                             data=json.dumps(data),
                             headers={'Content-Type': 'application/json',
                                      'Accept': 'application/octet-stream'})

    assert 200 == response.status_code
    assert 512 * 1024 ** 2 <= int(response.headers['Content-Length'])
