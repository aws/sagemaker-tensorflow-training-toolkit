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
import re
import shutil
import subprocess
import boto3
import container_support as cs
import google.protobuf.json_format as json_format
import os

from grpc import StatusCode, RpcError
from grpc.framework.interfaces.face.face import AbortionError
from tensorflow.core.framework import tensor_pb2
from tf_container import proxy_client
from six import StringIO
import csv
from container_support.serving import UnsupportedContentTypeError, UnsupportedAcceptTypeError, \
                                      JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, \
                                      OCTET_STREAM_CONTENT_TYPE, ANY_CONTENT_TYPE
from tf_container.run import logger
import time

DEFAULT_TF_SERVING_PORT = 9000
GENERIC_MODEL_NAME = "generic_model"
TF_SERVING_MAXIMUM_LOAD_MODEL_TIME_IN_SECONDS = 60 * 15


def export_saved_model(checkpoint_dir, model_path, s3=boto3.client('s3', region_name=os.environ.get('AWS_REGION'))):
    if checkpoint_dir.startswith('s3://'):
        bucket_name, key_prefix = cs.parse_s3_url(checkpoint_dir)
        prefix = os.path.join(key_prefix, 'export', 'Servo')

        try:
            contents = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]
            saved_model_path_array = [x['Key'] for x in contents if x['Key'].endswith('saved_model.pb')]

            if len(saved_model_path_array) == 0:
                logger.info("Failed to download saved model. File does not exist in {}".format(checkpoint_dir))
                return
        except KeyError as e:
            logger.error("Failed to download saved model. File does not exist in {}".format(checkpoint_dir))
            raise e
        # Select most recent saved_model.pb
        saved_model_path = saved_model_path_array[-1]
        saved_model_base_path = os.path.dirname(saved_model_path)

        prefixes = key_prefix.split('/')
        folders = saved_model_path.split('/')[len(prefixes):-1]
        path_to_save_model = os.path.join(model_path, *folders)

        def file_filter(x): return x['Key'].startswith(saved_model_base_path) and not x['Key'].endswith("/")
        paths_to_copy = [x['Key'] for x in contents if file_filter(x)]

        for key in paths_to_copy:
            target = re.sub(r"^"+saved_model_base_path, path_to_save_model, key)
            _makedirs_for_file(target)
            s3.download_file(bucket_name, key, target)
        logger.info("Downloaded saved model at {}".format(path_to_save_model))
    else:
        if os.path.exists(checkpoint_dir):
            _recursive_copy(checkpoint_dir, model_path)
        else:
            logger.error("Failed to copy saved model. File does not exist in {}".format(checkpoint_dir))


def _makedirs_for_file(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def _recursive_copy(src, dst):
    for root, dirs, files in os.walk(src):
        root = os.path.relpath(root, src)
        current_path = os.path.join(src, root)
        target_path = os.path.join(dst, root)

        for file in files:
            shutil.copy(os.path.join(current_path, file), os.path.join(target_path, file))
        for dir in dirs:
            new_dir = os.path.join(target_path, dir)
            if not os.path.exists(new_dir):
                os.mkdir(os.path.join(target_path, dir))


def transformer(user_module):
    env = cs.HostingEnvironment()

    port = int(cs.Server.next_safe_port(env.port_range)) if env.port_range else DEFAULT_TF_SERVING_PORT

    grpc_proxy_client = proxy_client.GRPCProxyClient(port)
    _wait_model_to_load(grpc_proxy_client, TF_SERVING_MAXIMUM_LOAD_MODEL_TIME_IN_SECONDS)

    return Transformer.from_module(user_module, grpc_proxy_client)


def load_dependencies():
    env = cs.HostingEnvironment()

    port = cs.Server.next_safe_port(env.port_range) if env.port_range else DEFAULT_TF_SERVING_PORT
    saved_model_path = os.path.join(env.model_dir, 'export/Servo')
    subprocess.Popen(['tensorflow_model_server',
                      '--port={}'.format(port),
                      '--model_name={}'.format(GENERIC_MODEL_NAME),
                      '--model_base_path={}'.format(saved_model_path)])


def _wait_model_to_load(grpc_proxy_client, max_seconds):
    """Wait TF Serving to load the model

    :param grpc_proxy_client: proxy client to make rpc call to TF Serving
    :param max_seconds: max number of seconds to wait
    """

    for i in range(max_seconds):
        try:
            grpc_proxy_client.cache_prediction_metadata()

            logger.info("TF Serving model successfully loaded")
            return
        except AbortionError as abort_err:
            if abort_err.code == StatusCode.UNAVAILABLE:
                _handle_rpc_exception(abort_err)
        # GRPC throws a _Rendezvous, which inherits from RpcError
        # _Rendezvous has a method for code instead of a parameter.
        # https://github.com/grpc/grpc/issues/9270
        except RpcError as rpc_error:
            if rpc_error.code() == StatusCode.UNAVAILABLE:
                _handle_rpc_exception(rpc_error)

    message = 'TF Serving failed to load the model under the maximum load time in seconds: {}'
    raise ValueError(message.format(max_seconds))


def _handle_rpc_exception(err):
    logger.info("Waiting for TF Serving to load the model due to {}"
                .format(err.__class__.__name__))
    time.sleep(1)


class Transformer(object):
    """A ``Transformer`` encapsulates the function(s) responsible
    for parsing incoming request data, passing it through an  and converting the result into something
    that can be returned as the body of and HTTP response.
    """

    def __init__(self, grpc_proxy_client, transform_fn=None, input_fn=None, output_fn=None):
        """Initialize a Transformer.

        :param transform_fn: a transformer function
        """
        self.proxy_client = grpc_proxy_client

        if transform_fn and (input_fn or output_fn):
            raise ValueError('transform_fn cannot be declared together with an input or output function.')

        if transform_fn:
            self.transform_fn = transform_fn
        else:
            input_fn = input_fn or self._default_input_fn
            output_fn = output_fn or self._default_output_fn

            self.transform_fn = self._build_transform_fn(input_fn, output_fn)

    @staticmethod
    def _parse_json_request(serialized_data):
        """
        json deserialization works in the following order:
            1 - tries to deserialize the payload as a tensor using google.protobuf.json_format.Parse(
                payload, tensor_pb2.TensorProto())
            2 - in case it fails uses common json.loads deserialization
        Args:
            serialized_data: (str) json data to be deserialized

        Returns:
            deserialized object
        """
        try:
            return json_format.Parse(serialized_data, tensor_pb2.TensorProto())
        except json_format.ParseError:
            return json.loads(serialized_data)

    @staticmethod
    def _parse_csv_request(serialized_data):
        """
        csv deserialization uses csv reader to create an array of floats (in absence of any other context information)
        The result is list of lists, each represents 1 line of the csv
        Args:
            serialized_data: (str) csv data to be deserialized

        Returns:
            list of lists of floats
        """
        csv_buff = StringIO(serialized_data)
        csv_to_parse = csv.reader(csv_buff, delimiter=',')

        # TODO csv is constructed as 2D arrays always, overall multi-point calls must be fixed
        row = next(csv_to_parse)
        full_array = [float(i) for i in row]

        return full_array

    def _build_transform_fn(self, input_fn, output_fn):
        """ Create a transformer function.
        :param input_fn: an input handler function
        :param output_fn: an output handler function
        :return:
        """

        def f(serialized_data, content_type, accepts):
            input = input_fn(serialized_data, content_type)
            prediction = self.predict_fn(input)
            output = output_fn(prediction, accepts)
            return output

        return f

    def predict_fn(self, data):
        """A default prediction function for TF models.

        :param data: deserialized data from the request
        :param content_type: content type of the request

        :return: Response received from the TF serving
        """

        return self.proxy_client.request(data)

    @staticmethod
    def _default_output_fn(data, accepts):
        if accepts in (JSON_CONTENT_TYPE, ANY_CONTENT_TYPE):
            return json_format.MessageToJson(data)
        if accepts == OCTET_STREAM_CONTENT_TYPE:
            return data.SerializeToString()

        raise UnsupportedAcceptTypeError('invalid accept type {}'.format(accepts))

    def _default_input_fn(self, serialized_data, content_type):
        if content_type == JSON_CONTENT_TYPE:
            return self._parse_json_request(serialized_data)
        if content_type == CSV_CONTENT_TYPE:
            return self._parse_csv_request(serialized_data)
        if content_type == OCTET_STREAM_CONTENT_TYPE:
            return self.proxy_client.parse_request(serialized_data)

        raise UnsupportedContentTypeError('Unsupported content-type {}'.format(content_type))

    @classmethod
    def from_module(cls, m, grpc_proxy_client):
        """Initialize a Transformer using functions supplied by the given module.

        If the module contains a ``transform_fn``, it will be used to handle incoming request
        data, execute the model prediction, and generation of response content.

        If the module does not contain a ``transform_fn``, then one will be assembled by chaining
        an ``input_fn``, ``predict_fn``, and ``output_fn``. Default handlers will be used
        for any of these that are not present in the supplied module.

        :param m: a python module
        :param grpc_proxy_client: grpc proxy client used to make rpc calls to TF serving
        :return: a configured Transformer object
        """

        if hasattr(m, 'transform_fn'):
            transform_fn = m.transform_fn
        else:
            input_fn = m.input_fn if hasattr(m, 'input_fn') else None
            output_fn = m.output_fn if hasattr(m, 'output_fn') else None

            return cls(grpc_proxy_client, input_fn=input_fn, output_fn=output_fn)

        return cls(grpc_proxy_client, transform_fn=transform_fn)

    def transform(self, data, content_type, accepts):
        """Transforms input data into a prediction result. The input data must
        be in a format compatible with the configured ``input_fn``. The output format
        will be determined by the ``output_fn``.

        Args:
            data: the content data
            content_type: (str) http content type
            accepts: (str) http accepts
        """
        return self.transform_fn(data, content_type, accepts), accepts
