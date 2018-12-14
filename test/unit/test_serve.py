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
from types import ModuleType

import pytest
from container_support.serving import UnsupportedAcceptTypeError, UnsupportedContentTypeError
from mock import Mock, call, patch

import tf_container.serve as serve

JSON_CONTENT_TYPE = "application/json"
FIRST_PORT = '1111'
LAST_PORT = '2222'
SAFE_PORT_RANGE = '{}-{}'.format(FIRST_PORT, LAST_PORT)


@pytest.fixture
def boto_session():
    session = Mock()

    return_value = {"Contents": [
        {'Key': 'test/1/'},
        {'Key': 'test/1/saved_model.pb'},
        {'Key': 'test/1/variables/variables.index'},
        {'Key': 'test/1/assets/vocabulary.txt'}
    ]}
    session.list_objects_v2 = Mock(return_value=return_value)
    session.download_file = Mock(return_value=None)
    return session


@patch('os.makedirs')
def test_export_saved_model_from_s3(makedirs, boto_session):
    serve.export_saved_model('s3://bucket/test', 'a/path', s3=boto_session)

    expected_boto_calls = [
        call('bucket', 'test/1/saved_model.pb', 'a/path/1/saved_model.pb'),
        call('bucket', 'test/1/variables/variables.index', 'a/path/1/variables/variables.index'),
        call('bucket', 'test/1/assets/vocabulary.txt', 'a/path/1/assets/vocabulary.txt')]

    expected_makedirs_calls = [
        call('a/path/1'),
        call('a/path/1/variables'),
        call('a/path/1/assets'),
    ]

    assert boto_session.download_file.mock_calls == expected_boto_calls
    assert makedirs.mock_calls == expected_makedirs_calls


@patch('os.path.exists')
@patch('os.makedirs')
def test_export_saved_model_from_filesystem(mock_exists, mock_makedirs):
    checkpoint_dir = 'a/dir'
    model_path = 'possible/another/dir'

    with patch('tf_container.serve._recursive_copy') as mock_copy:
        serve.export_saved_model(checkpoint_dir, model_path)
        mock_copy.assert_called_once_with(checkpoint_dir, model_path)


@pytest.fixture()
def mod():
    m = ModuleType('mod')
    yield m


@patch('google.protobuf.json_format.MessageToJson', side_effect=json.dumps)
def test_transformer_provides_default_transformer_fn(message):
    grpc_proxy_client = Mock()

    def _request(data):
        return data * 2

    grpc_proxy_client.request.side_effect = _request

    transformer = serve.Transformer(grpc_proxy_client=grpc_proxy_client)
    result = transformer.transform("[1,2,3]", "application/json", "application/json")

    assert result == ('[1, 2, 3, 1, 2, 3]', 'application/json')


def test_transformer_from_module_allows_users_to_provide_their_own_transform_fn(mod):
    def _transform_fn(data, content_type, accepts):
        return """
        transform_fn:
            content-type {},
            data {},
            accepts {}""".format(content_type, data, accepts)

    mod.transform_fn = _transform_fn

    transformer = serve.Transformer.from_module(mod, grpc_proxy_client=Mock())
    result = transformer.transform("my data", "application/json", "application/octet-stream")

    assert result[0] == """
        transform_fn:
            content-type application/json,
            data my data,
            accepts application/octet-stream"""

    assert result[1] == "application/octet-stream"


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_transformer_from_module_allows_users_to_provide_their_own_input_fns(proxy_client, mod):
    mod.input_fn = mock_input_fn
    mod.output_fn = mock_output_fn

    client = proxy_client()
    client.request.side_effect = mock_predict_fn

    transformer = serve.Transformer.from_module(mod, client)

    result = transformer.transform("my_data", "application/json", "application/octet-stream")

    assert result[0] == """
        output_fn:
            accept application/octet-stream,
            data
        predict_fn:
            data
        input_fn:
            content-type application/json,
            data my_data"""

    assert result[1] == "application/octet-stream"


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_transformer_from_module_separate_fn_csv(proxy_client, mod):
    client = proxy_client()
    client.request.side_effect = mock_predict_fn
    mod.input_fn = mock_input_fn
    mod.output_fn = mock_output_fn

    transformer = serve.Transformer.from_module(mod, client)

    result = transformer.transform("1,2,3\r\n", "text/csv", "text/csv")

    assert result[0] == """
        output_fn:
            accept text/csv,
            data
        predict_fn:
            data
        input_fn:
            content-type text/csv,
            data 1,2,3"""

    assert result[1] == "text/csv"


def mock_predict_fn(data):
    return """
        predict_fn:
            data
        {}""".format(data.strip())


def mock_input_fn(data, content_type):
    return """
        input_fn:
            content-type {},
            data {}""".format(content_type, data)


def mock_output_fn(data, content_type):
    return """
        output_fn:
            accept {},
            data
        {}""".format(content_type, data.strip())


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_transformer_from_module_separate_fn_protobuf(proxy_client, mod):
    client = proxy_client()
    client.request.side_effect = mock_predict_fn
    mod.input_fn = mock_input_fn
    mod.output_fn = mock_output_fn

    transformer = serve.Transformer.from_module(mod, client)

    result = transformer.transform("data", "application/octet-stream", "application/json")

    assert result[0] == """
        output_fn:
            accept application/json,
            data
        predict_fn:
            data
        input_fn:
            content-type application/octet-stream,
            data data"""

    assert result[1] == "application/json"


def test_transformer_from_module_default_fns(mod):
    transformer = serve.Transformer.from_module(mod, grpc_proxy_client=Mock())

    assert hasattr(transformer, 'transform_fn')


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_transformer_method(proxy_client):
    with patch('os.environ') as env:
        env['SAGEMAKER_PROGRAM'] = 'script.py'
        env['SAGEMAKER_SUBMIT_DIRECTORY'] = 's3://what/ever'

        user_module = Mock(spec=[])
        serve.Transformer.from_module = Mock()

        serve.transformer(user_module)

        serve.Transformer.from_module.assert_called_once_with(user_module, proxy_client())


@patch('subprocess.Popen')
@patch('container_support.HostingEnvironment')
def test_load_dependencies_with_default_port(hosting_env, popen):
    with patch('os.environ') as env:
        env['SAGEMAKER_PROGRAM'] = 'script.py'
        env['SAGEMAKER_SUBMIT_DIRECTORY'] = 's3://what/ever'

        hosting_env.return_value.port_range = None
        hosting_env.return_value.model_dir = '/opt/ml/model'

        serve.Transformer.from_module = Mock()
        serve.load_dependencies()

        popen.assert_called_with(['tensorflow_model_server',
                                  '--port=9000',
                                  '--model_name=generic_model',
                                  '--model_base_path=/opt/ml/model/export/Servo'])


@patch('subprocess.Popen')
@patch('container_support.HostingEnvironment')
def test_load_dependencies_with_safe_port(hosting_env, popen):
    with patch('os.environ') as env:
        env['SAGEMAKER_PROGRAM'] = 'script.py'
        env['SAGEMAKER_SUBMIT_DIRECTORY'] = 's3://what/ever'

        hosting_env.return_value.port_range = SAFE_PORT_RANGE
        hosting_env.return_value.model_dir = '/opt/ml/model'

        serve.Transformer.from_module = Mock()
        serve.load_dependencies()

        popen.assert_called_with(['tensorflow_model_server',
                                  '--port={}'.format(FIRST_PORT),
                                  '--model_name=generic_model',
                                  '--model_base_path=/opt/ml/model/export/Servo'])


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_wait_model_to_load(proxy_client):
    client = proxy_client()

    serve._wait_model_to_load(client, 10)
    client.cache_prediction_metadata.assert_called_once_with()


def test_transformer_default_output_fn_unsupported_type():
    accept_type = 'fake/accept-type'

    with pytest.raises(UnsupportedAcceptTypeError):
        serve.Transformer._default_output_fn(None, accept_type)


def test_transformer_default_input_fn_unsupported_type():
    content_type = 'fake/content-type'

    with pytest.raises(UnsupportedContentTypeError):
        serve.Transformer(None)._default_input_fn(None, content_type)


class DummyTransformer(object):
    def transform(self, content, mimetype):
        if content.startswith('500'):
            raise Exception("Dummy error")
        return 'transform -> {},{}'.format(content, mimetype)
