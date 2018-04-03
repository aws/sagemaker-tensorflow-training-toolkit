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
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS
from tensorflow_serving.apis import classification_pb2


def test_pb_request():
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = "generic_model"
    request.model_spec.signature_name = DEFAULT_SERVING_SIGNATURE_DEF_KEY
    example = request.input.example_list.examples.add()

    data = [4.9, 2.5, 4.5, 1.7]
    example.features.feature[PREDICT_INPUTS].float_list.value.extend(data)

    serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=request.SerializeToString(),
                                      headers={'Content-type': 'application/octet-stream',
                                               'Accept': 'application/octet-stream'}).content

    classification_response = classification_pb2.ClassificationResponse()
    classification_response.ParseFromString(serialized_output)

    classifications_classes = classification_response.result.classifications[0].classes
    assert len(classifications_classes) == 3
    for c in classifications_classes:
        assert c.score < 1
        assert c.score > 0


def test_json_request():
    data = [4.9, 2.5, 4.5, 1.7]

    serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=json.dumps(data),
                                      headers={'Content-type': 'application/json',
                                               'Accept': 'application/json'}).content
    prediction_result = json.loads(serialized_output)

    classes = prediction_result['result']['classifications'][0]['classes']
    assert len(classes) == 3
    assert classes[0].keys() == [u'score', u'label']


def test_csv_request():
    data = "4.9,2.5,4.5,1.7"

    serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=data,
                                      headers={'Content-type': 'text/csv',
                                               'Accept': 'application/json'}).content
    prediction_result = json.loads(serialized_output)

    classes = prediction_result['result']['classifications'][0]['classes']
    assert len(classes) == 3
    assert classes[0].keys() == [u'score', u'label']
