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
