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

import boto3
from sagemaker.tensorflow import TensorFlow
from six.moves.urllib.parse import urlparse

from sagemaker_tensorflow_container.training import SAGEMAKER_MPI_ENABLED, \
    SAGEMAKER_PARAMETER_SERVER_ENABLED


def test_mnist(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_type=instance_type,
                           train_instance_count=1,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           framework_version='1.11.0',
                           py_version='py3',
                           base_job_name='test-sagemaker-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data'),
        key_prefix='scriptmode/mnist')
    estimator.fit(inputs)
    model_s3_url = estimator.create_model().model_data
    _assert_s3_file_exists(model_s3_url)


def test_distributed_mnist_no_ps(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'distributed_mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           framework_version='1.11.0',
                           py_version='py3',
                           base_job_name='test-tf-sm-distributed-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist-distributed')
    estimator.fit(inputs)
    model_s3_url = estimator.create_model().model_data
    _assert_s3_file_exists(model_s3_url)


def test_distributed_mnist_ps(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'distributed_mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           hyperparameters={SAGEMAKER_PARAMETER_SERVER_ENABLED: True},
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           framework_version='1.11.0',
                           py_version='py3',
                           base_job_name='test-tf-sm-distributed-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist-distributed')
    estimator.fit(inputs)
    _assert_s3_file_exists(os.path.join(estimator.model_dir, 'graph.pbtxt'))
    _assert_s3_file_exists(os.path.join(estimator.model_dir, 'model.ckpt-0.index'))
    _assert_s3_file_exists(os.path.join(estimator.model_dir, 'model.ckpt-0.meta'))


def test_distributed_mnist_horovod(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'horovod_mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           hyperparameters={SAGEMAKER_MPI_ENABLED: True},
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           framework_version='1.11.0',
                           py_version='py3',
                           base_job_name='test-tf-sm-horovod-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist-distributed')
    estimator.fit(inputs)

def _assert_s3_file_exists(s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3')
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()
