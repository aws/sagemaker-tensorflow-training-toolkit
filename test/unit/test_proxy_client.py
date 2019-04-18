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
import os

import pytest
from google import protobuf
from mock import MagicMock, patch, ANY
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2

from tf_container.proxy_client import GRPCProxyClient, MAX_GRPC_MESSAGE_SIZE

REGRESSION = 'tensorflow/serving/regress'
INFERENCE = 'tensorflow/serving/inference'
CLASSIFY = 'tensorflow/serving/classify'
PREDICT = 'tensorflow/serving/predict'

DEFAULT_PORT = 9000
INPUT_TENSOR_NAME = 'inputs'


@pytest.fixture()
def proxy_client():
    proxy_client = GRPCProxyClient(DEFAULT_PORT, input_tensor_name=INPUT_TENSOR_NAME,
                                   signature_name='serving_default')
    proxy_client.input_type_map['sometype'] = 'somedtype'
    proxy_client.prediction_service_stub = MagicMock()

    return proxy_client


class PredictRequest(object):
    def __init__(self):
        self.model_spec = MagicMock()
        self.inputs = {INPUT_TENSOR_NAME: MagicMock()}


class ClassificationRequest(object):
    def __init__(self):
        self.mock = MagicMock()
        self.input = self.mock
        self.model_spec = self.mock
        self.input.example_list.examples = []


class Example(object):
    def __init__(self, features):
        self.features = features


class Features(object):
    def __init__(self, feature):
        self.feature = feature


class FeatureList(object):
    def __init__(self, value):
        self.value = value


class Feature(object):
    def __init__(self, int64_list=FeatureList([]), bytes_list=FeatureList([]),
                 float_list=FeatureList([])):
        self.int64_list = int64_list
        self.bytes_list = bytes_list
        self.float_list = float_list


class TensorProto(object):
    def __init__(self, data):
        self.data = data


@patch.dict(os.environ, {'SAGEMAKER_TFS_GRPC_REQUEST_TIMEOUT': '300.0',
                         'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_user_supplied_custom_tfs_timeout_with_ei():
    client = GRPCProxyClient(DEFAULT_PORT)

    assert client.request_timeout == 300.0


@patch.dict(os.environ, {'SAGEMAKER_TFS_GRPC_REQUEST_TIMEOUT': '300.0'}, clear=True)
def test_user_supplied_custom_tfs_timeout():
    client = GRPCProxyClient(DEFAULT_PORT)

    assert client.request_timeout == 300.0


@patch.dict(os.environ, {'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT': 'true'}, clear=True)
def test_default_tfs_ei_timeout():
    client = GRPCProxyClient(DEFAULT_PORT)

    assert client.request_timeout == 30


@patch('tensorflow_serving.apis.predict_pb2.PredictRequest')
def test_parse_request_predict(mock_request, proxy_client):
    proxy_client.prediction_type = PREDICT

    patch_metadata_request_with(PREDICT)
    proxy_client.parse_request('serialized_data')

    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


@patch('tensorflow_serving.apis.classification_pb2.ClassificationRequest')
def test_parse_request_classification(mock_request, proxy_client):
    proxy_client.prediction_type = CLASSIFY

    patch_metadata_request_with(CLASSIFY)
    proxy_client.parse_request('serialized_data')

    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


@patch('tensorflow_serving.apis.inference_pb2.MultiInferenceRequest')
def test_parse_request_inference(mock_request, proxy_client):
    proxy_client.prediction_type = INFERENCE

    patch_metadata_request_with(INFERENCE)
    proxy_client.parse_request('serialized_data')

    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


@patch('tensorflow_serving.apis.regression_pb2.RegressionRequest')
def test_parse_request_regression(mock_request, proxy_client):
    proxy_client.prediction_type = REGRESSION

    patch_metadata_request_with(REGRESSION)
    proxy_client.parse_request('serialized_data')

    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


def test_request_predict(proxy_client):
    proxy_client.prediction_type = PREDICT

    patch_metadata_request_with(PREDICT)

    predict_fn_mock = MagicMock()
    proxy_client.request_fn_map[PREDICT] = predict_fn_mock

    proxy_client.request('my_data')

    predict_fn_mock.assert_called_once_with('my_data')


def test_request_classification(proxy_client):
    proxy_client.prediction_type = CLASSIFY

    patch_metadata_request_with(CLASSIFY)

    mock = MagicMock()
    proxy_client.request_fn_map[CLASSIFY] = mock

    proxy_client.request('my_data')

    mock.assert_called_once_with('my_data')


def test_request_not_implemented(proxy_client):
    with pytest.raises(NotImplementedError):
        patch_metadata_request_with(INFERENCE)
        proxy_client.prediction_type = INFERENCE

        proxy_client.request('my_data')

    with pytest.raises(NotImplementedError):
        patch_metadata_request_with(REGRESSION)
        proxy_client.prediction_type = REGRESSION

        proxy_client.request('my_data')


@patch('tensorflow_serving.apis.predict_pb2.PredictRequest', new=PredictRequest)
@patch('tf_container.proxy_client.make_tensor_proto')
def test_predict_with_tensor_proto(make_tensor_proto, proxy_client):
    tensor_proto = TensorProto('/42-sagemaker')

    make_tensor_proto.return_value = tensor_proto

    prediction = proxy_client.predict(tensor_proto)

    predict_fn = proxy_client.prediction_service_stub.Predict
    predict_fn.assert_called_once()

    predict_request_attribute = predict_fn.call_args[0][0]

    assert predict_request_attribute.model_spec.name == 'generic_model'
    assert predict_request_attribute.model_spec.signature_name == 'serving_default'
    predict_request_attribute.inputs[INPUT_TENSOR_NAME].CopyFrom.assert_called_once_with(tensor_proto)

    assert prediction == predict_fn.return_value


@patch('tensorflow_serving.apis.predict_pb2.PredictRequest', new=PredictRequest)
@patch('tf_container.proxy_client.make_tensor_proto')
def test_predict_with_dict(make_tensor_proto, proxy_client):
    tensor_proto = TensorProto('/42-sagemaker')
    make_tensor_proto.return_value = tensor_proto
    prediction = proxy_client.predict({INPUT_TENSOR_NAME: tensor_proto})

    predict_fn = proxy_client.prediction_service_stub.Predict
    predict_fn.assert_called_once()

    predict_request_attribute = predict_fn.call_args[0][0]

    assert predict_request_attribute.model_spec.name == 'generic_model'
    assert predict_request_attribute.model_spec.signature_name == 'serving_default'
    predict_request_attribute.inputs[INPUT_TENSOR_NAME].CopyFrom.assert_called_once_with(tensor_proto)

    assert prediction == predict_fn.return_value


@patch('tensorflow_serving.apis.predict_pb2.PredictRequest', new=PredictRequest)
@patch('tf_container.proxy_client.make_tensor_proto')
def test_predict_with_predict_request(make_tensor_proto, proxy_client):
    request = PredictRequest()
    prediction = proxy_client.predict(request)

    predict_fn = proxy_client.prediction_service_stub.Predict
    predict_fn.assert_called_once()
    predict_fn.assert_called_once_with(request, proxy_client.request_timeout)

    assert prediction == predict_fn.return_value


@patch('tf_container.proxy_client.make_tensor_proto', side_effect=Exception('tensor proto failed!'))
def test_predict_with_invalid_payload(make_tensor_proto, proxy_client):
    data = complex('1+2j')

    with pytest.raises(ValueError) as error:
        proxy_client.predict(data)

    assert 'Unable to convert value to TensorProto' in str(error)


@patch('tf_container.proxy_client.make_tensor_proto', return_value='MyTensorProto')
def test_predict_create_input_map_with_dict_of_lists(make_tensor_proto, proxy_client):
    data = {'mytensor': [1, 2, 3]}

    result = proxy_client._create_input_map(data)
    assert result == {'mytensor': 'MyTensorProto'}
    make_tensor_proto.assert_called_once()


@patch('tensorflow_serving.apis.classification_pb2.ClassificationRequest', new=ClassificationRequest)
def test_classification_with_classification_request(proxy_client):
    request = ClassificationRequest()

    prediction = proxy_client.classification(request)

    classification_fn = proxy_client.prediction_service_stub.Classify
    classification_fn.assert_called_once_with(request, proxy_client.request_timeout)

    assert prediction == classification_fn.return_value


def test_classification_with_int_list(proxy_client):
    proxy_client.classification([1, 2, 3, 0])

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == []
    assert feature[INPUT_TENSOR_NAME].int64_list.value == [1, 2, 3, 0]
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == []


def test_classification_with_bytes_list(proxy_client):
    bytes = ['fnenfionk4235g', 'faf']
    proxy_client.classification(bytes)

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == []
    assert feature[INPUT_TENSOR_NAME].int64_list.value == []
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == bytes


def test_classification_with_float_list(proxy_client):
    data = [3.4000000953674316, 0.0, 0.0]
    proxy_client.classification(data)

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == data
    assert feature[INPUT_TENSOR_NAME].int64_list.value == []
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == []


def test_classification_with_int(proxy_client):
    proxy_client.classification(1)

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == []
    assert feature[INPUT_TENSOR_NAME].int64_list.value == [1]
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == []


def test_classification_with_bytes(proxy_client):
    bytes = 'fnenfionk4235g'
    proxy_client.classification(bytes)

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == []
    assert feature[INPUT_TENSOR_NAME].int64_list.value == []
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == [bytes]


def test_classification_with_float(proxy_client):
    data = 3.4000000953674316
    proxy_client.classification(data)
    proxy_client.prediction_service_stub.Classify.assert_called_once()

    classification_request = _get_classification_request(proxy_client)
    feature = _get_feature(classification_request)

    assert feature[INPUT_TENSOR_NAME].float_list.value == [data]
    assert feature[INPUT_TENSOR_NAME].int64_list.value == []
    assert feature[INPUT_TENSOR_NAME].bytes_list.value == []


def test_classification_with_invalid_payload(proxy_client):
    data = complex('1+2j')

    with pytest.raises(ValueError) as error:
        proxy_client.classification(data)

    assert 'Unsupported request data format' in str(error)


def test_classification_protobuf(proxy_client):
    request = MagicMock()
    proxy_client.classification(request)
    proxy_client.prediction_service_stub.Classify.assert_called_once()


@patch('tensorflow_serving.apis.get_model_metadata_pb2.SignatureDefMap')
@patch('tensorflow_serving.apis.get_model_metadata_pb2.GetModelMetadataRequest')
@patch('tensorflow_serving.apis.prediction_service_pb2_grpc.PredictionServiceStub')
@patch('grpc.insecure_channel')
def test_cache_prediction_metadata(channel, stub, request, signature_def_map, proxy_client):
    proxy_client.cache_prediction_metadata()

    channel.assert_called_once_with('localhost:{}'.format(DEFAULT_PORT), options=[
              ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_SIZE),
              ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_SIZE)])

    stub = prediction_service_pb2_grpc.PredictionServiceStub
    stub.assert_called_once_with(channel())

    request = get_model_metadata_pb2.GetModelMetadataRequest
    request.assert_called_once()

    stub().GetModelMetadata.assert_called_once_with(request(), 10.0)

    assert proxy_client.prediction_service_stub == stub.return_value


def patch_metadata_request_with(method_name):
    protobuf.json_format.MessageToJson.return_value = json.dumps({
        'metadata': {
            'signature_def': {
                'signatureDef': {
                    'serving_default': {
                        'methodName': method_name
                    }
                }
            }
        }
    })


def _get_classification_request(proxy_client):
    predict_fn = proxy_client.prediction_service_stub.Classify
    predict_fn.assert_called_once_with(ANY, proxy_client.request_timeout)
    first_call = predict_fn.call_args[0]
    classification_request = first_call[0]
    return classification_request


def _get_feature(classification_request):
    examples = classification_request.input.example_list.examples
    assert len(examples) == 1
    feature = examples[0].features.feature
    return feature
