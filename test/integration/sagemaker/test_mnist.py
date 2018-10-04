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
from sagemaker.tensorflow import TensorFlow


@pytest.mark.skip_gpu
def test_mnist_cpu(sagemaker_session, ecr_image, instance_type):
    _run_minist_training(sagemaker_session, ecr_image, instance_type)


@pytest.mark.skip_cpu
def test_mnist_gpu(sagemaker_session, ecr_image, instance_type):
    _run_minist_training(sagemaker_session, ecr_image, instance_type)


def _run_minist_training(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           training_steps=1,
                           evaluation_steps=1,
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           base_job_name='test-sagemaker-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data'),
        key_prefix='scriptmode/mnist')
    estimator.fit(inputs)
