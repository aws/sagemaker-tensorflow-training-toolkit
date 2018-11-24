# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import numpy as np

from sagemaker.tensorflow import TensorFlow, serving

from test.integration import RESOURCE_PATH

logging.basicConfig(level=logging.DEBUG)


def test_keras_training(sagemaker_local_session, docker_image, tmpdir):
    entry_point = os.path.join(RESOURCE_PATH, 'keras_inception.py')
    output_path = 'file://{}'.format(tmpdir)

    estimator = TensorFlow(
        entry_point=entry_point,
        role='SageMakerRole',
        train_instance_count=1,
        train_instance_type='local',
        sagemaker_session=sagemaker_local_session,
        model_dir='/opt/ml/model',
        output_path=output_path,
        framework_version='1.11.0',
        py_version='py3')

    estimator.fit()

    model = serving.Model(model_data=output_path, role='SageMakerRole',
                          framework_version='1.11.0',
                          sagemaker_session=sagemaker_local_session)

    predictor = model.deploy(initial_instance_count=1, instance_type='local')

    assert predictor.predict(np.random.randn(4, 4, 4, 2) * 255)
