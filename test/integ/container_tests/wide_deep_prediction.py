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
