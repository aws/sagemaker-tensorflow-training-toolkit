import numpy as np
from grpc.beta import implementations
from tensorflow import make_tensor_proto
from tensorflow.core.example import example_pb2, feature_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import predict_pb2, classification_pb2, inference_pb2, regression_pb2
from tensorflow_serving.apis import prediction_service_pb2

REGRESSION = 'tensorflow/serving/regression'
CLASSIFY = 'tensorflow/serving/classify'
INFERENCE = 'tensorflow/serving/inference'
PREDICT = 'tensorflow/serving/predict'
GENERIC_MODEL_NAME = "generic_model"


class GRPCProxyClient(object):
    def __init__(self, tf_serving_port, host='localhost', request_timeout=10.0, model_name=GENERIC_MODEL_NAME,
                 input_tensor_name=PREDICT_INPUTS, signature_name=DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        self.tf_serving_port = tf_serving_port
        self.host = host
        self.request_timeout = request_timeout
        self.model_name = model_name
        self.input_tensor_name = input_tensor_name
        self.signature_name = signature_name
        self.request_fn_map = {PREDICT: self.predict,
                               CLASSIFY: self.classification,
                               # TODO: implement inference and regression tf serving apis
                               INFERENCE: self._raise_not_implemented_exception,
                               REGRESSION: self._raise_not_implemented_exception,
                               }
        self.prediction_type = None
        self.input_type_map = {}

    def parse_request(self, serialized_data):
        request_fn_map = {
            PREDICT: lambda: predict_pb2.PredictRequest(),
            INFERENCE: lambda: inference_pb2.MultiInferenceRequest(),
            CLASSIFY: lambda: classification_pb2.ClassificationRequest(),
            REGRESSION: lambda: regression_pb2.RegressionRequest()
        }

        request = request_fn_map[self.prediction_type]()
        request.ParseFromString(serialized_data)

        return request

    def request(self, data):
        request_fn = self.request_fn_map[self.prediction_type]
        return request_fn(data)

    def cache_prediction_metadata(self):
        channel = implementations.insecure_channel(self.host, self.tf_serving_port)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        request = get_model_metadata_pb2.GetModelMetadataRequest()

        request.model_spec.name = self.model_name
        request.metadata_field.append('signature_def')
        result = stub.GetModelMetadata(request, self.request_timeout)

        signature_def = result.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)

        serving_default = signature_map.ListFields()[0][1]['serving_default']
        serving_inputs = serving_default.inputs

        self.input_type_map = {key: serving_inputs[key].dtype for key in serving_inputs.keys()}
        self.prediction_type = serving_default.method_name

    def predict(self, data):

        channel = implementations.insecure_channel(self.host, self.tf_serving_port)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        request = self._create_predict_request(data)

        result = stub.Predict(request, self.request_timeout)

        return result

    def _create_predict_request(self, data):
        # Send request
        # See prediction_service.proto for gRPC request/response details.

        if isinstance(data, predict_pb2.PredictRequest):
            return data

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        input_map = self._create_input_map(data)

        for k, v in input_map.items():
            try:
                request.inputs[k].CopyFrom(v)
            except:
                raise ValueError("""Unsupported request data format: {}.
                Valid formats: tensor_pb2.TensorProto and predict_pb2.PredictRequest""".format(type(data)))

        return request

    def classification(self, data):
        channel = implementations.insecure_channel(self.host, self.tf_serving_port)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        request = self._create_classification_request(data)

        result = stub.Classify(request, self.request_timeout)

        return result

    def _create_classification_request(self, data):
        if isinstance(data, classification_pb2.ClassificationRequest):
            return data

        request = classification_pb2.ClassificationRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        features_map_list = self._create_features_map_list(data)

        examples = [_create_tf_example(features_map) for features_map in features_map_list]

        request.input.example_list.examples.extend(examples)

        return request

    def _create_features_map_list(self, data):
        """
        Parses the input data and returns a [dict<string, iterable>] which will be used to create the tf examples.
        If the input data is not a dict, a dictionary will be created with the default predict key PREDICT_INPUTS

        Examples:
            input                                   => output
            {'age': 39., 'workclass': 'Private'}    => [{'age': 39., 'workclass': 'Private'}]
            [{'age': 39., 'workclass': 'Private'}]  => [{'age': 39., 'workclass': 'Private'}]
            [1, 2, 'string']                        => [{PREDICT_INPUTS: [1, 2, 'string']}]
            42                                      => [{PREDICT_INPUTS: [42]}]

        Args:
            data: request data. Can be an instance of float, int, str, map, or any iterable object.


        Returns: a dict[string, iterable] that will be used to create the tf example

        """
        if isinstance(data, dict):
            return [data]
        if hasattr(data, '__iter__'):
            if all(isinstance(x, dict) for x in data):
                return data
            return [{self.input_tensor_name: data}]
        return [{self.input_tensor_name: [data]}]

    def _raise_not_implemented_exception(self, data):
        raise NotImplementedError('This prediction service type is not supported py SageMaker yet')

    def _create_input_map(self, data):
        """
        Parses the input data and returns a dict<string, TensorProto> which will be used to create the predict request.
        If the input data is not a dict, a dictionary will be created with the default predict key PREDICT_INPUTS

        input.

        Examples:
            input                                   => output
            {'inputs': tensor_proto}                => {'inputs': tensor_proto}
            tensor_proto                            => {PREDICT_INPUTS: tensor_proto}
            [1,2,3]                                 => {PREDICT_INPUTS: tensor_proto(1,2,3)}
        Args:
            data: request data. Can be any instance of dict<string, tensor_proto>, tensor_proto or any array like data.

        Returns:
            dict<string, tensor_proto>


        """
        msg = """Unsupported request data format: {}.
Valid formats: tensor_pb2.TensorProto, dict<string,  tensor_pb2.TensorProto> and predict_pb2.PredictRequest"""

        if isinstance(data, dict):
            if all(isinstance(v, tensor_pb2.TensorProto) for k, v in data.items()):
                return data
            raise ValueError(msg.format(data))

        if isinstance(data, tensor_pb2.TensorProto):
            return {self.input_tensor_name: data}

        try:
            # TODO: tensorflow container supports prediction requests with ONLY one tensor as input
            input_type = self.input_type_map.values()[0]
            ndarray = np.asarray(data)
            tensor_proto = make_tensor_proto(values=ndarray, dtype=input_type, shape=ndarray.shape)
            return {self.input_tensor_name: tensor_proto}
        except:
            raise ValueError(msg.format(data))


def _create_tf_example(feature_map):
    """
    Creates a tf example protobuf message given a feature map. The protobuf message is defined here
        https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/input.proto#L19
    Args:
        feature_map: a list of feature maps

    Returns:
        a tf.train.Example including the features
    """

    def _create_feature(feature_list):
        float_list = []
        int64_list = []
        bytes_list = []

        for x in feature_list:
            if type(x) == int:
                int64_list.append(x)
            elif type(x) == str:
                bytes_list.append(x)
            elif type(x) == float:
                float_list.append(x)
            else:
                message = """Unsupported request data format: {}.
                                Valid formats: float, int, str any object that implements __iter__
                                               or classification_pb2.ClassificationRequest"""
                raise ValueError(message.format(feature_list))

        return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=int64_list),
                                   bytes_list=feature_pb2.BytesList(value=bytes_list),
                                   float_list=feature_pb2.FloatList(value=float_list))

    features = {k: _create_feature(v) for k, v in feature_map.items()}
    return example_pb2.Example(features=feature_pb2.Features(feature=features))
