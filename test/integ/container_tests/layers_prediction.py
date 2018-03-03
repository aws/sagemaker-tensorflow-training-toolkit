import json

import numpy as np
import requests
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS
from tensorflow_serving.apis import predict_pb2


def test_pb_request():
    data = [x for x in xrange(784)]
    tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "generic_model"
    request.model_spec.signature_name = DEFAULT_SERVING_SIGNATURE_DEF_KEY

    request.inputs[PREDICT_INPUTS].CopyFrom(tensor_proto)

    serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=request.SerializeToString(),
                                      headers={
                                          'Content-type': 'application/octet-stream',
                                          'Accept': 'application/octet-stream'
                                      }).content

    predict_response = predict_pb2.PredictResponse()
    predict_response.ParseFromString(serialized_output)

    probabilities = predict_response.outputs['probabilities']
    assert len(probabilities.float_val) == 10
    for v in probabilities.float_val:
        assert v <= 1.
        assert v >= 0.


def test_json_request():
    data = [x for x in xrange(784)]
    tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)

    url = "http://localhost:8080/invocations"
    serialized_output = requests.post(url,
                                      MessageToJson(tensor_proto),
                                      headers={'Content-type': 'application/json'}).content

    prediction_result = json.loads(serialized_output)

    assert len(prediction_result['outputs']['probabilities']['floatVal']) == 10
