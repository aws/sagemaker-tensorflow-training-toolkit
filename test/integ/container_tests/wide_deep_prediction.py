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


def test_json_request():
    data = {'age': 42., 'workclass': 'Private', 'education': 'Doctorate', 'education_num': 16.,
                     'marital_status': 'Married-civ-spouse', 'occupation': 'Prof-specialty', 'relationship': 'Husband',
                     'capital_gain': 0., 'capital_loss': 0., 'hours_per_week': 45.}

    response = requests.post("http://localhost:8080/invocations",
                                      data=json.dumps(data),
                                      headers={'Content-type': 'application/json',
                                               'Accept': 'application/json'})

    serialized_output = response.content

    prediction_result = json.loads(serialized_output)

    classes = prediction_result['result']['classifications'][0]['classes']
    assert len(classes) == 2
    assert classes[0].keys() == [u'score', u'label']
