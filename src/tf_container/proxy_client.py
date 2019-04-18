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
import os

import numpy as np
from google.protobuf import json_format
import grpc
from tensorflow import make_tensor_proto
from tensorflow.core.example import example_pb2, feature_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY, \
    PREDICT_INPUTS
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import predict_pb2, classification_pb2, inference_pb2, regression_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tf_container.run import logger as _logger

INFERENCE_ACCELERATOR_PRESENT_ENV = 'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT'
TF_SERVING_GRPC_REQUEST_TIMEOUT_ENV = 'SAGEMAKER_TFS_GRPC_REQUEST_TIMEOUT'

MAX_GRPC_MESSAGE_SIZE = 1024 ** 3 * 2 - 1  # 2GB - 1
DEFAULT_GRPC_REQUEST_TIMEOUT_FOR_INFERENCE_ACCELERATOR = 30.0

REGRESSION = 'tensorflow/serving/regress'
CLASSIFY = 'tensorflow/serving/classify'
INFERENCE = 'tensorflow/serving/inference'
PREDICT = 'tensorflow/serving/predict'
GENERIC_MODEL_NAME = "generic_model"


class GRPCProxyClient(object):
    def __init__(self, tf_serving_port, host='localhost', request_timeout=10.0,
                 model_name=GENERIC_MODEL_NAME,
                 input_tensor_name=PREDICT_INPUTS,
                 signature_name=DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        if os.environ.get(TF_SERVING_GRPC_REQUEST_TIMEOUT_ENV):
            request_timeout = float(os.environ.get(TF_SERVING_GRPC_REQUEST_TIMEOUT_ENV))
        elif os.environ.get(INFERENCE_ACCELERATOR_PRESENT_ENV) == 'true':
            request_timeout = DEFAULT_GRPC_REQUEST_TIMEOUT_FOR_INFERENCE_ACCELERATOR

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
        self.prediction_service_stub = None
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
        channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.tf_serving_port),
            options=[
              ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_SIZE),
              ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_SIZE)])
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = get_model_metadata_pb2.GetModelMetadataRequest()

        request.model_spec.name = self.model_name
        request.metadata_field.append('signature_def')
        result = stub.GetModelMetadata(request, self.request_timeout)

        _logger.info('---------------------------Model Spec---------------------------')
        _logger.info(json_format.MessageToJson(result))
        _logger.info('----------------------------------------------------------------')

        signature_def = result.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)

        serving_default = signature_map.ListFields()[0][1]['serving_default']
        serving_inputs = serving_default.inputs

        self.input_type_map = {key: serving_inputs[key].dtype for key in serving_inputs.keys()}
        self.prediction_type = serving_default.method_name
        self.prediction_service_stub = stub

    def predict(self, data):
        request = self._create_predict_request(data)
        result = self.prediction_service_stub.Predict(request, self.request_timeout)
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
        request = self._create_classification_request(data)
        result = self.prediction_service_stub.Classify(request, self.request_timeout)
        return result

    def _create_classification_request(self, data):
        if isinstance(data, classification_pb2.ClassificationRequest):
            return data

        request = classification_pb2.ClassificationRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        feature_dict_list = self._create_feature_dict_list(data)

        examples = [_create_tf_example(feature_dict) for feature_dict in feature_dict_list]

        request.input.example_list.examples.extend(examples)

        return request

    def _create_feature_dict_list(self, data):
        """
        Parses the input data and returns a [dict<string, iterable>] which will be used to create the tf examples.
        If the input data is not a dict, a dictionary will be created with the default key PREDICT_INPUTS.
        Used on the code path for creating ClassificationRequests.

        Examples:
            input                                   => output
            {'age': 39., 'workclass': 'Private'}    => [{'age': 39., 'workclass': 'Private'}]
            [{'age': 39., 'workclass': 'Private'}]  => [{'age': 39., 'workclass': 'Private'}]
            [{'age': 39., 'workclass': 'Private'}, {'age': 39., 'workclass':'Public'}]
                                                    => [{'age': 39., 'workclass': 'Private'},
                                                        {'age': 39., 'workclass': 'Public'}]
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
        raise NotImplementedError('This prediction service type is not supported by SageMaker yet')

    def _create_input_map(self, data):
        """
        Parses the input data and returns a dict<string, TensorProto> which will be used to create the PredictRequest.
        If the input data is not a dict, a dictionary will be created with the default predict key PREDICT_INPUTS

        input.

        Examples:
            input                                   => output
            -------------------------------------------------
            tensor_proto                            => {PREDICT_INPUTS: tensor_proto}
            {'custom_tensor_name': tensor_proto}    => {'custom_tensor_name': TensorProto}
            [1,2,3]                                 => {PREDICT_INPUTS: TensorProto(1,2,3)}
            {'custom_tensor_name': [1, 2, 3]}       => {'custom_tensor_name': TensorProto(1,2,3)}
        Args:
            data: request data. Can be any of: ndarray-like, TensorProto, dict<str, TensorProto>, dict<str, ndarray-like>

        Returns:
            dict<string, tensor_proto>


        """
        if isinstance(data, dict):
            return {k: self._value_to_tensor(v) for k, v in data.items()}

        # When input data is not a dict, no tensor names are given, so use default
        return {self.input_tensor_name: self._value_to_tensor(data)}

    def _value_to_tensor(self, value):
        """Converts the given value to a tensor_pb2.TensorProto. Used on code path for creating PredictRequests."""
        if isinstance(value, tensor_pb2.TensorProto):
            return value

        msg = """Unable to convert value to TensorProto: {}. 
        Valid formats: tensor_pb2.TensorProto, list, numpy.ndarray"""
        try:
            # TODO: tensorflow container supports prediction requests with ONLY one tensor as input
            input_type = self.input_type_map.values()[0]
            ndarray = np.asarray(value)
            return make_tensor_proto(values=ndarray, dtype=input_type, shape=ndarray.shape)
        except Exception:
            raise ValueError(msg.format(value))


def _create_tf_example(feature_dict):
    """
    Creates a tf example protobuf message given a feature dict. The protobuf message is defined here
        https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/input.proto#L19
    Args:
        feature_dict (dict of str -> feature): feature can be any of the following:
          int, strings, unicode object, float, or list of any of the previous types.

    Returns:
        a tf.train.Example including the features
    """

    def _create_feature(feature):
        feature_list = feature if isinstance(feature, list) else [feature]

        # Each feature can be exactly one kind:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto#L76

        feature_type = type(feature_list[0])
        if feature_type == int:
            return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=feature_list))
        elif feature_type == str:
            return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=feature_list))
        elif feature_type == unicode:
            return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=map(lambda x: str(x), feature_list)))
        elif feature_type == float:
            return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=feature_list))
        else:
            message = """Unsupported request data format: {}, {}.
                            Valid formats: float, int, str any object that implements __iter__
                                           or classification_pb2.ClassificationRequest"""
            raise ValueError(message.format(feature, type(feature)))

    features = {k: _create_feature(v) for k, v in feature_dict.items()}
    return example_pb2.Example(features=feature_pb2.Features(feature=features))
