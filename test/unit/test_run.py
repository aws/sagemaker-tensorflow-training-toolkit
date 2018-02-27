import pytest
from mock import MagicMock, patch, call, mock_open
from test.unit.utils import mock_import_modules


@pytest.fixture
def trainer():
    return MagicMock()


@pytest.fixture
def run():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'google.protobuf.json_format',
        'tensorflow.contrib.learn.python.learn.utils',
        'tensorflow.core.example',
        'logging',
        'tensorflow.python.estimator',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    import tf_container.run as run
    yield run
    patcher.stop()


@patch('__builtin__.open', mock_open())
def test_train_and_log_exceptions(trainer, run):
    run.train_and_log_exceptions(trainer, 'my/output/path')

    trainer.train.assert_called_once()
    open.assert_called_with('my/output/path/success', 'w')
    open().write.assert_called_once_with('Done')


@patch('__builtin__.open', mock_open())
@patch('traceback.format_exc')
def test_train_and_log_exceptions_writes_errors_to_file(trc, trainer, run):
    trainer.train.side_effect = RuntimeError()

    with pytest.raises(RuntimeError):
        run.train_and_log_exceptions(trainer, 'my/output/path')

    trainer.train.assert_called_once()
    trc.assert_called_once()

    open.assert_called_with('my/output/path/failure', 'w')
    open().write.assert_called_once()
