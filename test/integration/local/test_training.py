# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest
from sagemaker.estimator import Framework
from sagemaker.tensorflow import TensorFlow

from test.integration.docker_utils import Container


@pytest.fixture
def py_full_version(py_version):
    if py_version == '2':
        return '2.7'
    else:
        return '3.6'


def test_py_versions(docker_image, processor, py_full_version):
    with Container(docker_image, processor) as c:
        output = c.execute_command(['python', '--version'])
        assert output.strip().startswith('Python {}'.format(py_full_version))


@pytest.mark.skip_gpu
def test_mnist_cpu(sagemaker_local_session, docker_image):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources', 'mnist')
    output_path = run_tf_training(script=os.path.join(resource_path, 'mnist.py'),
                                  instance_type='local',
                                  instance_count=1,
                                  sagemaker_local_session=sagemaker_local_session,
                                  docker_image=docker_image,
                                  training_data_path='file://{}'.format(
                                      os.path.join(resource_path, 'data')))
    assert os.path.exists(os.path.join(output_path, 'my_model.h5')), 'model file not found'


@pytest.mark.skip_cpu
def test_gpu(sagemaker_local_session, docker_image):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    run_tf_training(script=os.path.join(resource_path, 'gpu_device_placement.py'),
                    instance_type='local_gpu',
                    instance_count=1,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
                    training_data_path='file://{}'.format(
                        os.path.join(resource_path, 'mnist', 'data')))


@pytest.mark.skip_gpu
def test_distributed_training_cpu(sagemaker_local_session, docker_image):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    run_tf_training(script=os.path.join(resource_path, 'mnist', 'distributed_mnist.py'),
                    instance_type='local',
                    instance_count=2,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
                    training_data_path='file://{}'.format(
                        os.path.join(resource_path, 'mnist', 'data-distributed')))


@pytest.mark.skip_gpu
def test_distributed_training_cpu_1_ps(sagemaker_local_session, docker_image):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    run_tf_training(script=os.path.join(resource_path, 'mnist', 'distributed_mnist.py'),
                    instance_type='local',
                    instance_count=2,
                    sagemaker_local_session=sagemaker_local_session,
                    docker_image=docker_image,
                    hyperparameters={'sagemaker_parameter_server_num': 1},
                    training_data_path='file://{}'.format(
                        os.path.join(resource_path, 'mnist', 'data-distributed')))


class ScriptModeTensorFlow(Framework):
    """This class is temporary until the final version of Script Mode is released.
    """

    __framework_name__ = "tensorflow-scriptmode-beta"

    create_model = TensorFlow.create_model

    def __init__(self, py_version='py', **kwargs):
        self.requirements_file = None
        self.py_version = py_version
        self.framework_version = 'some version'
        super(ScriptModeTensorFlow, self).__init__(**kwargs)


def run_tf_training(script, instance_type, instance_count,
                    sagemaker_local_session,
                    docker_image, training_data_path,
                    hyperparameters={}):
    estimator = ScriptModeTensorFlow(entry_point=script,
                                     role='SageMakerRole',
                                     train_instance_count=instance_count,
                                     train_instance_type=instance_type,
                                     sagemaker_session=sagemaker_local_session,
                                     image_name=docker_image,
                                     hyperparameters=hyperparameters,
                                     base_job_name='test-tf')

    estimator.fit(training_data_path)
    model = estimator.create_model()
    return model.model_data
