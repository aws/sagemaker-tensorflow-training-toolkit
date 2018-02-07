import json
import pytest
from mock import MagicMock, patch, ANY

from test.utils import mock_import_modules

REGRESSION = 'tensorflow/serving/regression'
INFERENCE = 'tensorflow/serving/inference'
CLASSIFY = 'tensorflow/serving/classify'
PREDICT = 'tensorflow/serving/predict'


@pytest.fixture()
def set_up():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'google.protobuf.json_format',
        'tensorflow.contrib.learn.python.learn.utils',
        'tensorflow.python.estimator',
        'tensorflow.core.example',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    from tf_container.proxy_client import GRPCProxyClient
    proxy_client = GRPCProxyClient(9000, input_tensor_name='inputs', signature_name='serving_default')

    yield mock, proxy_client
    patcher.stop()


@pytest.fixture()
def set_up_requests(set_up):
    mock, proxy_client = set_up
    mock.tensor_pb2.TensorProto = TensorProto
    mock.predict_pb2.PredictRequest = PredictRequest
    mock.classification_pb2.ClassificationRequest = ClassificationRequest
    mock.example_pb2.Example = Example
    mock.feature_pb2.Features = Features
    mock.feature_pb2.Feature = Feature
    mock.feature_pb2.Int64List = FeatureList
    mock.feature_pb2.BytesList = FeatureList
    mock.feature_pb2.FloatList = FeatureList


class PredictRequest(object):
    def __init__(self):
        self.model_spec = MagicMock()
        self.inputs = {'inputs': MagicMock()}


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


class Feature(object):
    def __init__(self, int64_list, bytes_list, float_list):
        self.int64_list = int64_list
        self.bytes_list = bytes_list
        self.float_list = float_list


class FeatureList(object):
    def __init__(self, value):
        self.value = value


class TensorProto(object):
    def __init__(self, data):
        self.data = data


def test_parse_request_predict(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = PREDICT

    patch_metadata_request_with(mock, PREDICT)
    proxy_client.parse_request('serialized_data')

    mock_request = mock.predict_pb2.PredictRequest
    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


def test_parse_request_classification(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = CLASSIFY

    patch_metadata_request_with(mock, CLASSIFY)
    proxy_client.parse_request('serialized_data')

    mock_request = mock.classification_pb2.ClassificationRequest
    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


def test_parse_request_inference(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = INFERENCE

    patch_metadata_request_with(mock, INFERENCE)
    proxy_client.parse_request('serialized_data')

    mock_request = mock.inference_pb2.MultiInferenceRequest
    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


def test_parse_request_regression(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = REGRESSION

    patch_metadata_request_with(mock, REGRESSION)
    proxy_client.parse_request('serialized_data')

    mock_request = mock.regression_pb2.RegressionRequest
    mock_request.assert_called_once()
    mock_request().ParseFromString.assert_called_once_with('serialized_data')


def test_request_predict(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = PREDICT

    patch_metadata_request_with(mock, PREDICT)

    predict_fn_mock = MagicMock()
    proxy_client.request_fn_map[PREDICT] = predict_fn_mock

    proxy_client.request('my_data')

    predict_fn_mock.assert_called_once_with('my_data')


def test_request_classification(set_up):
    mock, proxy_client = set_up
    proxy_client.prediction_type = CLASSIFY

    patch_metadata_request_with(mock, CLASSIFY)

    mock = MagicMock()
    proxy_client.request_fn_map[CLASSIFY] = mock

    proxy_client.request('my_data')

    mock.assert_called_once_with('my_data')


def test_request_not_implemented(set_up):
    mock, proxy_client = set_up

    with pytest.raises(NotImplementedError):
        patch_metadata_request_with(mock, INFERENCE)
        proxy_client.prediction_type = INFERENCE

        proxy_client.request('my_data')

    with pytest.raises(NotImplementedError):
        patch_metadata_request_with(mock, REGRESSION)
        proxy_client.prediction_type = REGRESSION

        proxy_client.request('my_data')


def test_predict_with_tensor_proto(set_up, set_up_requests):
    mock, proxy_client = set_up

    tensor_proto = TensorProto('/42-sagemaker')
    prediction = proxy_client.predict(tensor_proto)

    create_stub = assert_prediction_service_was_called(mock)

    predict_fn = create_stub.return_value.Predict
    predict_fn.assert_called_once_with(ANY, proxy_client.request_timeout)

    first_call = predict_fn.call_args[0]
    predict_request_attribute = first_call[0]

    assert predict_request_attribute.model_spec.name == 'generic_model'
    assert predict_request_attribute.model_spec.signature_name == 'serving_default'
    predict_request_attribute.inputs['inputs'].CopyFrom.assert_called_once_with(tensor_proto)

    assert prediction == predict_fn.return_value


def test_predict_with_dict(set_up, set_up_requests):
    mock, proxy_client = set_up

    tensor_proto = TensorProto('/42-sagemaker')
    prediction = proxy_client.predict({'inputs': tensor_proto})

    create_stub = assert_prediction_service_was_called(mock)

    predict_fn = create_stub.return_value.Predict
    predict_fn.assert_called_once_with(ANY, proxy_client.request_timeout)

    first_call = predict_fn.call_args[0]
    predict_request_attribute = first_call[0]

    assert predict_request_attribute.model_spec.name == 'generic_model'
    assert predict_request_attribute.model_spec.signature_name == 'serving_default'
    predict_request_attribute.inputs['inputs'].CopyFrom.assert_called_once_with(tensor_proto)

    assert prediction == predict_fn.return_value


def test_predict_with_predict_request(set_up, set_up_requests):
    mock, proxy_client = set_up

    request = PredictRequest()
    prediction = proxy_client.predict(request)

    create_stub = assert_prediction_service_was_called(mock)

    predict_fn = create_stub.return_value.Predict
    predict_fn.assert_called_once_with(request, proxy_client.request_timeout)

    assert prediction == predict_fn.return_value


def test_predict_with_invalid_payload(set_up, set_up_requests):
    mock, proxy_client = set_up

    data = complex('1+2j')

    with pytest.raises(ValueError) as error:
        proxy_client.predict(data)

    assert 'Unsupported request data format' in str(error)


def test_classification_with_classification_request(set_up, set_up_requests):
    mock, proxy_client = set_up

    request = ClassificationRequest()

    prediction = proxy_client.classification(request)

    create_stub = assert_prediction_service_was_called(mock)

    classification_fn = create_stub.return_value.Classify
    classification_fn.assert_called_once_with(request, proxy_client.request_timeout)

    assert prediction == classification_fn.return_value


def test_classification_with_int_list(set_up, set_up_requests):
    mock, proxy_client = set_up

    proxy_client.classification([1, 2, 3, 0])

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == []
    assert feature['inputs'].int64_list.value == [1, 2, 3, 0]
    assert feature['inputs'].bytes_list.value == []


def test_classification_with_dict(set_up, set_up_requests):
    mock, proxy_client = set_up

    proxy_client.classification({'my_tensor_name': [1, 2., 'private', 0]})

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['my_tensor_name'].float_list.value == [2.]
    assert feature['my_tensor_name'].int64_list.value == [1, 0]
    assert feature['my_tensor_name'].bytes_list.value == ['private']


def test_classification_with_bytes_list(set_up, set_up_requests):
    mock, proxy_client = set_up

    bytes = ['fnenfionk4235g', 'faf']
    proxy_client.classification(bytes)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == []
    assert feature['inputs'].int64_list.value == []
    assert feature['inputs'].bytes_list.value == bytes


def test_classification_with_float_list(set_up, set_up_requests):
    mock, proxy_client = set_up

    data = [3.4, 0.0, 0.0]
    proxy_client.classification(data)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == data
    assert feature['inputs'].int64_list.value == []
    assert feature['inputs'].bytes_list.value == []


def test_classification_with_int(set_up, set_up_requests):
    mock, proxy_client = set_up

    proxy_client.classification(1)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == []
    assert feature['inputs'].int64_list.value == [1]
    assert feature['inputs'].bytes_list.value == []


def test_classification_with_bytes(set_up, set_up_requests):
    mock, proxy_client = set_up

    bytes = 'fnenfionk4235g'
    proxy_client.classification(bytes)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == []
    assert feature['inputs'].int64_list.value == []
    assert feature['inputs'].bytes_list.value == [bytes]


def test_classification_with_float(set_up, set_up_requests):
    mock, proxy_client = set_up

    data = 3.4
    proxy_client.classification(data)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == [data]
    assert feature['inputs'].int64_list.value == []
    assert feature['inputs'].bytes_list.value == []


def test_classification_with_mixed_elements(set_up, set_up_requests):
    mock, proxy_client = set_up

    data = [3.4, 0, 'aaargh']
    proxy_client.classification(data)

    create_stub = assert_prediction_service_was_called(mock)
    classification_request = _get_classification_request(create_stub, proxy_client)
    feature = _get_feature(classification_request)

    assert feature['inputs'].float_list.value == [3.4]
    assert feature['inputs'].int64_list.value == [0]
    assert feature['inputs'].bytes_list.value == ['aaargh']


def test_classification_with_invalid_payload(set_up, set_up_requests):
    mock, proxy_client = set_up

    data = complex('1+2j')

    with pytest.raises(ValueError) as error:
        proxy_client.classification(data)

    assert 'Unsupported request data format' in str(error)


def test_classification_protobuf(set_up, set_up_requests):
    mock, proxy_client = set_up

    request = MagicMock()
    proxy_client.classification(request)

    assert_prediction_service_was_called(mock)


def test_cache_prediction_metadata(set_up):
    mock, proxy_client = set_up

    proxy_client.cache_prediction_metadata()

    channel = mock.implementations.insecure_channel
    channel.assert_called_once_with('localhost', 9000)

    stub = mock.prediction_service_pb2.beta_create_PredictionService_stub
    stub.assert_called_once_with(channel())

    request = mock.get_model_metadata_pb2.GetModelMetadataRequest
    request.assert_called_once()

    stub().GetModelMetadata.assert_called_once_with(request(), 10.0)


def assert_prediction_service_was_called(mock):
    insecure_channel = mock.implementations.insecure_channel.return_value
    create_stub = mock.prediction_service_pb2.beta_create_PredictionService_stub
    create_stub.assert_called_with(insecure_channel)
    return create_stub


def patch_metadata_request_with(mock, method_name):
    mock.protobuf.json_format.MessageToJson.return_value = json.dumps({
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


def _get_classification_request(create_stub, proxy_client):
    predict_fn = create_stub.return_value.Classify
    predict_fn.assert_called_once_with(ANY, proxy_client.request_timeout)
    first_call = predict_fn.call_args[0]
    classification_request = first_call[0]
    return classification_request


def _get_feature(classification_request):
    examples = classification_request.input.example_list.examples
    assert len(examples) == 1
    feature = examples[0].features.feature
    return feature