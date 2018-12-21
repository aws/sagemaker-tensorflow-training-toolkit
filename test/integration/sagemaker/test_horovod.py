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

from sagemaker.tensorflow import serving, TensorFlow

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


def test_distributed_training_horovod(sagemaker_session,
                                      sagemaker_local_session,
                                      instance_type,
                                      ecr_image,
                                      tmpdir):

    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, 'mnist', 'horovod_mnist.py'),
        role='SageMakerRole',
        train_instance_type=instance_type,
        train_instance_count=2,
        image_name=ecr_image,
        framework_version='1.12',
        py_version='py3',
        script_mode=True,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': '-verbose',
                         'sagemaker_mpi_num_of_processes_per_host': 1})

    estimator.fit()

    model = serving.Model(model_data=estimator.model_data,
                          role='SageMakerRole',
                          framework_version='1.12.0',
                          sagemaker_session=sagemaker_local_session)

    predictor = model.deploy(initial_instance_count=1, instance_type='local')
    assert predictor.predict([[0] * 28] * 28)

    predictor.delete_endpoint()
