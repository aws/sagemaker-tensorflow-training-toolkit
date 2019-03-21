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

import logging
import os

import numpy as np
import pytest
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.utils import sagemaker_timestamp

from test.functional import EI_SUPPORTED_REGIONS
from test.integ.conftest import SCRIPT_PATH
from test.resources.python_sdk.timeout import timeout_and_delete_endpoint_by_name

logger = logging.getLogger(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)
logging.getLogger('session.py').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.fixture(autouse=True)
def skip_if_non_supported_ei_region(region):
    if region not in EI_SUPPORTED_REGIONS:
        pytest.skip('EI is not supported in {}'.format(region))


@pytest.fixture
def pretrained_model_data(region):
    return 's3://sagemaker-sample-data-{}/tensorflow/model/resnet/resnet_50_v2_fp32_NCHW.tar.gz'.format(region)


# based on https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_using_elastic_inference_with_your_own_model/tensorflow_pretrained_model_elastic_inference.ipynb
@pytest.mark.skip_if_non_supported_ei_region
@pytest.mark.skip_if_no_accelerator
def test_deploy_elastic_inference_with_pretrained_model(pretrained_model_data, docker_image_uri, sagemaker_session, instance_type, accelerator_type):
    resource_path = os.path.join(SCRIPT_PATH, '../resources')
    endpoint_name = 'test-tf-ei-deploy-model-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session,
                                             minutes=20):
        tensorflow_model = TensorFlowModel(model_data=pretrained_model_data,
                                           entry_point='default_entry_point.py',
                                           source_dir=resource_path,
                                           role='SageMakerRole',
                                           image=docker_image_uri,
                                           sagemaker_session=sagemaker_session)

        logger.info('deploying model to endpoint: {}'.format(endpoint_name))
        predictor = tensorflow_model.deploy(initial_instance_count=1,
                                            instance_type=instance_type,
                                            accelerator_type=accelerator_type,
                                            endpoint_name=endpoint_name)

        random_input = np.random.rand(1, 1, 3, 3)

        predict_response = predictor.predict({'input': random_input.tolist()})
        assert predict_response['outputs']['probabilities']
