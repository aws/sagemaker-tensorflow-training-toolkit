# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import subprocess
import sys

CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'
DLC_AWS_ID = '763104351884'


def build_image(framework_version, dockerfile, image_uri, region, cwd='.'):
    _check_call('python setup.py sdist')

    if 'dlc' in dockerfile:
        ecr_login(region, DLC_AWS_ID)

    dockerfile_location = os.path.join('test-toolkit', 'docker', framework_version, dockerfile)

    subprocess.check_call(
        ['docker', 'build', '-t', image_uri, '-f', dockerfile_location, '--build-arg',
         'region={}'.format(region), cwd], cwd=cwd)
    print('created image {}'.format(image_uri))
    return image_uri


def push_image(ecr_image, region, aws_id):
    ecr_login(region, aws_id)
    _check_call('docker push {}'.format(ecr_image))


def ecr_login(region, aws_id):
    login = _check_call('aws ecr get-login --registry-ids {} '.format(aws_id)
                        + '--no-include-email --region {}'.format(region))
    _check_call(login.decode('utf-8').rstrip('\n'))


def _check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd)
    return subprocess.check_output(cmd, *popenargs, **kwargs)


def _print_cmd(cmd):
    print('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    sys.stdout.flush()
