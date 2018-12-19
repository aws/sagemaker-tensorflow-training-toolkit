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

import pytest
from test.integ.docker_utils import Container


@pytest.fixture
def required_versions(framework_version):
    if framework_version == '1.4.1':
        return ['tensorflow-serving-api==1.4.0',
                'tensorflow==1.4.1']
    elif framework_version == '1.5.0':
        return ['tensorflow-serving-api==1.5.0',
                'tensorflow==1.5.0']
    # We released using TensorFlow Serving 1.5.0 for tf 1.6, due to not finding this
    # fix in time before launch: https://github.com/tensorflow/serving/issues/819
    elif framework_version == '1.6.0':
        return ['tensorflow-serving-api==1.5.0',
                'tensorflow==1.6.0']
    elif framework_version == '1.7.0':
        return ['tensorflow-serving-api==1.7.0',
                'tensorflow==1.7.0']
    # TODO: upgrade to serving 1.8.0 (see tfserving-1.8 branch)
    elif framework_version == '1.8.0':
        return ['tensorflow-serving-api==1.7.0',
                'tensorflow==1.8.0']
    elif framework_version == '1.9.0':
        return ['tensorflow-serving-api==1.7.0',
                'tensorflow==1.9.0']
    elif framework_version == '1.10.0':
        return ['tensorflow-serving-api==1.7.0',
                'tensorflow==1.10.0']
    elif framework_version == '1.11.0':
        return ['tensorflow-serving-api==1.11.0',
                'tensorflow==1.11.0']
    elif framework_version == '1.12.0':
        return ['tensorflow-serving-api==1.12.0',
                'tensorflow==1.12.0']
    else:
        raise ValueError("invalid internal test config")


def test_framework_versions(docker_image, processor, required_versions):
    with Container(docker_image, processor) as c:
        output = c.execute_command(['pip', 'freeze'])
        lines = output.splitlines()
        result = sorted([v for v in lines if v in required_versions])

        assert required_versions == result
