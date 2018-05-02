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

import os

from sagemaker import fw_utils

from test.integ.docker_utils import train, HostingContainer
from test.integ.utils import copy_resource, create_config_files, file_exists
from test.integ.conftest import SCRIPT_PATH


def test_estimator_classification(docker_image, sagemaker_session, opt_ml, processor):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/iris')

    copy_resource(resource_path, opt_ml, 'data', 'input/data')

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                bucket=sagemaker_session.default_bucket(),
                                s3_key_prefix='test_job',
                                script='iris.py',
                                directory=os.path.join(resource_path, 'code'))

    additional_hyperparameters = dict(training_steps=1, evaluation_steps=1, sagemaker_requirements='requirements.txt')
    create_config_files('iris.py', s3_source_archive.s3_prefix, opt_ml,
                          additional_hyperparameters)
    os.makedirs(os.path.join(opt_ml, 'model'))

    train(docker_image, opt_ml, processor)

    assert file_exists(opt_ml, 'model/export/Servo'), 'model was not exported'
    assert file_exists(opt_ml, 'model/checkpoint'), 'checkpoint was not created'
    assert file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not file_exists(opt_ml, 'output/failure'), 'Failure happened'

    with HostingContainer(opt_ml=opt_ml, image=docker_image, script_name='iris.py',
                          processor=processor, requirements_file='requirements.txt') as c:
        c.execute_pytest('test/integ/container_tests/estimator_classification.py')

        modules = c.execute_command(['pip', 'freeze'])
        assert 'beautifulsoup4==4.6.0' in modules
