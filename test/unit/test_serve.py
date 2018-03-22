import json

import pytest
from mock import Mock, call, patch
from test.unit.utils import mock_import_modules
from types import ModuleType

JSON_CONTENT_TYPE = "application/json"


@pytest.fixture(scope="module")
def modules():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'google.protobuf.json_format',
        'tensorflow.core.example',
        'tensorflow.contrib.learn.python.learn.utils',
        'tensorflow.contrib.training.HParams',
        'tensorflow.python.estimator',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    yield mock
    patcher.stop()


@pytest.fixture(scope="module")
def serve(modules):
    # Import module here to utilize mocked dependencies in modules()
    import tf_container.serve as serve
    yield serve


@pytest.fixture
def boto_session():
    session = Mock()

    return_value = {"Contents": [{'Key': 'test/1/saved_model.pb'}, {'Key': 'test/1/variables/variables.index'}]}
    session.list_objects_v2 = Mock(return_value=return_value)
    session.download_file = Mock(return_value=None)
    return session


@patch('os.makedirs')
def test_export_saved_model_from_s3(makedirs, boto_session, serve):
    serve.export_saved_model('s3://bucket/test', 'a/path', s3=boto_session)

    first_call = call('bucket', 'test/1/saved_model.pb', 'a/path/1/saved_model.pb')
    second_call = call('bucket', 'test/1/variables/variables.index', 'a/path/1/variables/variables.index')

    calls = [first_call, second_call]

    boto_session.download_file.assert_has_calls(calls)


@patch('os.path.exists')
@patch('os.makedirs')
def test_export_saved_model_from_filesystem(mock_exists, mock_makedirs, serve):
    checkpoint_dir = 'a/dir'
    model_path = 'possible/another/dir'

    with patch('shutil.copy2') as mock_copy:
        serve.export_saved_model(checkpoint_dir, model_path)
        mock_copy.assert_called_once_with(checkpoint_dir, model_path)


@pytest.fixture()
def mod():
    m = ModuleType('mod')
    yield m


@pytest.fixture()
def json_format(modules):
    def _json_format_parse(serialized_data, tensor_proto):
        return json.loads(serialized_data)

    modules.protobuf.json_format.Parse.side_effect = _json_format_parse

    def _json_format_message_to_json(data):
        return json.dumps(data)

    modules.protobuf.json_format.MessageToJson.side_effect = _json_format_message_to_json
    return modules.protobuf.json_format


def test_transformer_provides_default_transformer_fn(serve, mod, json_format):
    grpc_proxy_client = Mock()

    def _request(data):
        return data * 2

    grpc_proxy_client.request.side_effect = _request

    transformer = serve.Transformer(grpc_proxy_client=grpc_proxy_client)
    result = transformer.transform("[1,2,3]", "application/json", "application/json")

    assert result == ('[1, 2, 3, 1, 2, 3]', 'application/json')


def test_transformer_from_module_allows_users_to_provide_their_own_transform_fn(serve, mod):
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
def test_transformer_from_module_allows_users_to_provide_their_own_input_fns(proxy_client, modules, serve, mod):
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
def test_transformer_from_module_separate_fn_csv(proxy_client, serve, mod):
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
def test_transformer_from_module_separate_fn_protobuf(proxy_client, modules, serve, mod):
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


def test_transformer_from_module_default_fns(serve, mod):
    transformer = serve.Transformer.from_module(mod, grpc_proxy_client=Mock())

    assert hasattr(transformer, 'transform_fn')


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_transformer_method(proxy_client, serve):
    with patch('os.environ') as env:
        env['SAGEMAKER_PROGRAM'] = 'script.py'
        env['SAGEMAKER_SUBMIT_DIRECTORY'] = 's3://what/ever'

        user_module = Mock(spec=[])
        serve.Transformer.from_module = Mock()

        serve.transformer(user_module)

        serve.Transformer.from_module.assert_called_once_with(user_module, proxy_client())


@patch('subprocess.Popen')
def test_load_dependencies(popen, serve):
    with patch('os.environ') as env:
        env['SAGEMAKER_PROGRAM'] = 'script.py'
        env['SAGEMAKER_SUBMIT_DIRECTORY'] = 's3://what/ever'

        serve.Transformer.from_module = Mock()
        serve.load_dependencies()

        popen.assert_called_with(['tensorflow_model_server',
                                  '--port=9000',
                                  '--model_name=generic_model',
                                  '--model_base_path=/opt/ml/model/export/Servo'])


@patch('tf_container.proxy_client.GRPCProxyClient')
def test_wait_model_to_load(proxy_client, serve):
    client = proxy_client()

    serve._wait_model_to_load(client, 10)
    client.cache_prediction_metadata.assert_called_once_with()


class DummyTransformer(object):
    def transform(self, content, mimetype):
        if content.startswith('500'):
            raise Exception("Dummy error")
        return 'transform -> {},{}'.format(content, mimetype)
