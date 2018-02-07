import pytest
from mock import patch, call, MagicMock, ANY
from test.utils import mock_import_modules

mock_script = {}
hosts = ['algo-1', 'algo-2', 'algo-3']
current_host = ['algo-1']
model_path = "a/mock/path"


@pytest.fixture(scope="module")
def modules():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.contrib.learn.python.learn.utils',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow.core.protobuf',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'tensorflow.contrib.learn',
        'google.protobuf.json_format',
        'tensorflow.python.estimator',
        'tensorflow.core.example',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    yield mock
    patcher.stop()


@pytest.fixture(scope="module")
def trainer(modules):
    import tf_container.trainer as trainer
    yield trainer


def test_build_tf_config_with_one_host(trainer):
    hosts = ['algo-1']
    current_host = 'algo-1'

    test_wrapper = trainer.Trainer(customer_script=mock_script,
                                   current_host=current_host,
                                   hosts=hosts,
                                   model_path=model_path)

    tf_config = test_wrapper.build_tf_config()

    expected_tf_config = {
        'environment': 'cloud',
        'cluster': {
            'master': ['algo-1:2222']
        },
        'task': {'index': 0, 'type': 'master'}
    }

    assert tf_config == expected_tf_config
    assert test_wrapper.task_type == 'master'


def test_build_tf_config_with_multiple_hosts(trainer):
    hosts = ['algo-1', 'algo-2', 'algo-3', 'algo-4']
    current_host = 'algo-3'

    test_wrapper = trainer.Trainer(customer_script=mock_script,
                                   current_host=current_host,
                                   hosts=hosts,
                                   model_path=model_path)

    tf_config = test_wrapper.build_tf_config()

    expected_tf_config = {
        'environment': 'cloud',
        'cluster': {
            'master': ['algo-1:2222'],
            'ps': ['algo-1:2223', 'algo-2:2223', 'algo-3:2223', 'algo-4:2223'],
            'worker': ['algo-2:2222', 'algo-3:2222', 'algo-4:2222']
        },
        'task': {'index': 1, 'type': 'worker'}
    }

    assert tf_config == expected_tf_config
    assert test_wrapper.task_type == 'worker'


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_configure_s3_file_system(os_env, botocore, boto_client, trainer):
    trainer.Trainer(customer_script=mock_script,
                    current_host=current_host,
                    hosts=hosts,
                    model_path='s3://my/s3/path')

    boto_client('s3').get_bucket_location.assert_called_once_with(Bucket='my')

    calls = [
        call('S3_USE_HTTPS', '1'),
        call('S3_REGION', boto_client('s3').get_bucket_location()['LocationConstraint'])
    ]

    os_env.__setitem__.assert_has_calls(calls, any_order=True)


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_trainer_keras_model_fn(os_environ, botocore, boto3, trainer, modules):
    '''
    this test ensures that customers functions model_fn, train_input_fn, eval_input_fn, and serving_input_fn are
    being invoked with the right params
    '''
    customer_script = MagicMock(spec=['keras_model_fn', 'train_input_fn', 'eval_input_fn', 'serving_input_fn'])

    _trainer = trainer.Trainer(customer_script=customer_script,
                               current_host=current_host,
                               hosts=hosts,
                               model_path='s3://my/s3/path',
                               customer_params={'training_steps': 10, 'num_gpu': 20},
                               training_path='mytrainingpath')

    modules.learn_runner.run.side_effect = lambda experiment_fn, training_path: experiment_fn(training_path)
    modules.Experiment.side_effect = execute_input_functions
    modules.saved_model_export_utils.make_export_strategy.side_effect = make_export_strategy_fn

    _trainer.train()

    expected_params = {'num_gpu': 20, 'min_eval_frequency': 1000, 'training_steps': 10, 'save_checkpoints_secs': 300}

    modules.keras.estimator.model_to_estimator.assert_called_with(
        config=modules.RunConfig(),
        keras_model=customer_script.keras_model_fn(),
    )

    modules.learn_runner.run.assert_called()
    modules.Experiment.assert_called()

    customer_script.train_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.eval_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.serving_input_fn.assert_called_with(expected_params)


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_trainer_model_fn(os_environ, botocore, boto3, trainer, modules):
    '''
    this test ensures that customers functions model_fn, train_input_fn, eval_input_fn, and serving_input_fn are
    being invoked with the right params
    '''
    customer_script = MagicMock(spec=['model_fn', 'train_input_fn', 'eval_input_fn', 'serving_input_fn'])

    _trainer = trainer.Trainer(customer_script=customer_script,
                               current_host=current_host,
                               hosts=hosts,
                               model_path='s3://my/s3/path',
                               customer_params={'training_steps': 10, 'num_gpu': 20},
                               training_path='mytrainingpath')

    modules.learn_runner.run.side_effect = lambda experiment_fn, training_path: experiment_fn(training_path)
    modules.Experiment.side_effect = execute_input_functions
    modules.saved_model_export_utils.make_export_strategy.side_effect = make_export_strategy_fn

    _trainer.train()

    expected_params = {'num_gpu': 20, 'min_eval_frequency': 1000, 'training_steps': 10, 'save_checkpoints_secs': 300}

    modules.estimator.Estimator.assert_called_with(
        config=modules.RunConfig(),
        model_fn=ANY,
        params=expected_params
    )

    modules.learn_runner.run.assert_called()
    modules.Experiment.assert_called()

    customer_script.train_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.eval_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.serving_input_fn.assert_called_with(expected_params)


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_trainer_experiment_params(os_environ, botocore, boto3, trainer, modules):
    '''
    this test ensures that customers functions model_fn, train_input_fn, eval_input_fn, and serving_input_fn are
    being invoked with the right params
    '''
    customer_script = MagicMock(spec=['model_fn', 'train_input_fn', 'eval_input_fn', 'serving_input_fn'])

    _trainer = trainer.Trainer(customer_script=customer_script,
                               current_host=current_host,
                               hosts=hosts,
                               model_path='s3://my/s3/path',
                               eval_steps=23,
                               customer_params={'min_eval_frequency': 2,
                                                'local_eval_frequency': 3,
                                                'eval_delay_secs': 7,
                                                'continuous_eval_throttle_secs': 25,
                                                'train_steps_per_iteration': 13},
                               training_path='mytrainingpath')

    modules.learn_runner.run.side_effect = lambda experiment_fn, training_path: experiment_fn(training_path)

    _trainer.train()

    modules.Experiment.assert_called_with(
        estimator=modules.estimator.Estimator(),
        train_input_fn=ANY,
        eval_input_fn=ANY,
        export_strategies=ANY,
        train_steps=ANY,
        eval_steps=23,
        min_eval_frequency=2,
        local_eval_frequency=3,
        eval_delay_secs=7,
        continuous_eval_throttle_secs=25,
        train_steps_per_iteration=13
    )


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_trainer_run_config_params(os_environ, botocore, boto3, trainer, modules):
    '''
    this test ensures that customers functions model_fn, train_input_fn, eval_input_fn, and serving_input_fn are
    being invoked with the right params
    '''
    customer_script = MagicMock(spec=['model_fn', 'train_input_fn', 'eval_input_fn', 'serving_input_fn'])

    _trainer = trainer.Trainer(customer_script=customer_script,
                               current_host=current_host,
                               hosts=hosts,
                               model_path='s3://my/s3/path',
                               eval_steps=23,
                               customer_params={'save_summary_steps': 1,
                                                'save_checkpoints_secs': 2,
                                                'save_checkpoints_steps': 3,
                                                'keep_checkpoint_max': 4,
                                                'keep_checkpoint_every_n_hours': 5,
                                                'log_step_count_steps': 6},
                               training_path='mytrainingpath')
    modules.learn_runner.run.side_effect = lambda experiment_fn, training_path: experiment_fn(training_path)

    _trainer.train()

    modules.RunConfig.assert_called_with(
        save_summary_steps=1,
        save_checkpoints_secs=2,
        save_checkpoints_steps=3,
        keep_checkpoint_max=4,
        keep_checkpoint_every_n_hours=5,
        log_step_count_steps=6,
        model_dir='s3://my/s3/path'
    )


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_train_estimator_fn(os_environ, botocore, boto3, trainer, modules):
    '''
    this test ensures that customers functions estimator_fn, train_input_fn, eval_input_fn, and serving_input_fn are
    being invoked with the right params
    '''
    customer_script = MagicMock(spec=['estimator_fn', 'train_input_fn', 'eval_input_fn', 'serving_input_fn'])

    _trainer = trainer.Trainer(customer_script=customer_script,
                               current_host=current_host,
                               hosts=hosts,
                               model_path='s3://my/s3/path',
                               customer_params={'training_steps': 10, 'num_gpu': 20},
                               training_path='mytrainingpath')

    modules.learn_runner.run.side_effect = lambda experiment_fn, training_path: experiment_fn(training_path)
    modules.Experiment.side_effect = execute_input_functions
    modules.saved_model_export_utils.make_export_strategy.side_effect = make_export_strategy_fn

    _trainer.train()

    modules.learn_runner.run.assert_called()
    modules.Experiment.assert_called()

    expected_params = {'num_gpu': 20, 'min_eval_frequency': 1000, 'training_steps': 10, 'save_checkpoints_secs': 300}
    customer_script.estimator_fn.assert_called_with(modules.RunConfig(), expected_params)
    customer_script.train_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.eval_input_fn.assert_called_with('mytrainingpath', expected_params)
    customer_script.serving_input_fn.assert_called_with(expected_params)


def execute_input_functions(estimator,
                            train_input_fn,
                            eval_input_fn,
                            **kwargs):
    train_input_fn()
    eval_input_fn()


def make_export_strategy_fn(serving_input_fn, default_output_alternative_key, exports_to_keep):
    serving_input_fn()
