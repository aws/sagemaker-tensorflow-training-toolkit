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

import pytest
from test.integ.conftest import SCRIPT_PATH
from test.integ.docker_utils import Container


@pytest.fixture(autouse=True)
def skip_cpu(request, processor):
    if request.node.get_closest_marker('skip_cpu') and processor == 'cpu':
        pytest.skip('Skipping because we are running cpu image.')


@pytest.mark.skip_cpu
def test_gpu_devices_availability(docker_image, processor):
    script = os.path.join(SCRIPT_PATH, '../resources/gpu_device_placement.py')

    try:
        with Container(docker_image, processor) as c:
            c.copy(script, '/')
            c.execute_command(['python', '/gpu_device_placement.py'])
    except ValueError:
        pytest.fail("Can not access GPUs from GPU image.\n" +
                    "Verify that image has tensorflow-gpu installed and runs on GPU instance.")
