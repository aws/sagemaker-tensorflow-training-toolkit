#  Copyright <YEAR> Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the 'License').
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the 'license' file accompanying this file. This file is distributed 
#  on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import pytest
from mock import patch, call, MagicMock, ANY
from test.unit.utils import mock_import_modules


@pytest.fixture(scope='module')
def modules():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow.core.protobuf',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'google.protobuf.json_format',
        'tensorflow.core.example',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    yield mock
    patcher.stop()


MOCK_SCRIPT = {}
HOSTS = ['algo-1', 'algo-2', 'algo-3']
CURRENT_HOST = ['algo-1']
MODEL_PATH = 'a/mock/path'
TRAIN_DIR = 'another/mock/path'
INPUT_CHANNELS = {'training': TRAIN_DIR}
HYPERPARAMETERS = {'strparam': 'strval', 'intparam': 789}

class EmptyModule(object):
    pass


@pytest.fixture(scope='module')
def trainer_module(modules):
    import tf_container.trainer
    yield tf_container.trainer


@pytest.fixture
def trainer(trainer_module):
    yield trainer_module.Trainer(customer_script=MOCK_SCRIPT,
                                 current_host=CURRENT_HOST,
                                 hosts=HOSTS,
                                 model_path=MODEL_PATH,
                                 input_channels=INPUT_CHANNELS.copy(),
                                 customer_params=HYPERPARAMETERS.copy())


def test_special_params_defaulting(trainer_module):
    trainer = trainer_module.Trainer(customer_script=MOCK_SCRIPT,
                                     current_host=CURRENT_HOST,
                                     hosts=HOSTS,
                                     model_path=MODEL_PATH)
    assert trainer.customer_params['save_checkpoints_secs'] == 300


def test_build_run_config(modules, trainer):
    trainer.customer_params['save_checkpoints_secs'] = 123

    conf = trainer._build_run_config()

    modules.estimator.RunConfig.assert_called_with(model_dir=MODEL_PATH, save_checkpoints_secs=123)
    assert modules.estimator.RunConfig.return_value == conf


def test_build_train_spec(modules, trainer):
    tensor_dict = {'inputs': ['faketensor']}
    labels = ['fakelabels']
    # We add some defaulted hyperparameters into the customer params.
    expected_hps = HYPERPARAMETERS.copy()
    expected_hps['save_checkpoints_secs'] = 300

    # Set up "customer script".
    def customer_train_input_fn(training_dir, hyperparameters):
        assert training_dir == TRAIN_DIR
        assert hyperparameters == expected_hps
        return tensor_dict, labels
    customer_script = EmptyModule()
    customer_script.train_input_fn = customer_train_input_fn

    trainer.train_steps = 987
    trainer.customer_script = customer_script

    spec = trainer._build_train_spec()

    modules.estimator.TrainSpec.assert_called_with(ANY, max_steps=987)
    assert modules.estimator.TrainSpec.return_value == spec
    # Assert that we passed a 0-arg function to TrainSpec as the train_input_fn, that when called,
    # Invokes the customer's train_input_fn with the correct training_dir and hyperparameters.
    train_input_fn = modules.estimator.TrainSpec.call_args[0][0]
    returned_dict, returned_labels = train_input_fn()
    assert (tensor_dict, labels) == (returned_dict, returned_labels)


def test_build_tf_config_with_one_host(trainer):
    trainer.hosts = ['algo-1']
    trainer.current_host = 'algo-1'

    tf_config = trainer.build_tf_config()

    expected_tf_config = {
        'environment': 'cloud',
        'cluster': {
            'master': ['algo-1:2222']
        },
        'task': {'index': 0, 'type': 'master'}
    }

    assert tf_config == expected_tf_config
    assert trainer.task_type == 'master'


def test_build_tf_config_with_multiple_hosts(trainer):
    trainer.hosts = ['algo-1', 'algo-2', 'algo-3', 'algo-4']
    trainer.current_host = 'algo-3'

    tf_config = trainer.build_tf_config()

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
    assert trainer.task_type == 'worker'


@patch('boto3.client')
@patch('botocore.session.get_session')
@patch('os.environ')
def test_configure_s3_file_system(os_env, botocore, boto_client, trainer_module):
    trainer_module.Trainer(customer_script=MOCK_SCRIPT,
                           current_host=CURRENT_HOST,
                           hosts=HOSTS,
                           model_path='s3://my/s3/path')

    boto_client('s3').get_bucket_location.assert_called_once_with(Bucket='my')

    calls = [
        call('S3_USE_HTTPS', '1'),
        call('S3_REGION', boto_client('s3').get_bucket_location()['LocationConstraint'])
    ]

    os_env.__setitem__.assert_has_calls(calls, any_order=True)