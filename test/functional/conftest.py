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

import boto3
import pytest
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)


def pytest_addoption(parser):
    parser.addoption('--aws-id')
    parser.addoption('--docker-base-name', default='preprod-tensorflow')
    parser.addoption('--instance-type')
    parser.addoption('--accelerator-type', default=None)
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default=TensorFlow.LATEST_VERSION)
    parser.addoption('--processor', default='cpu', choices=['gpu', 'cpu'])
    parser.addoption('--tag')


@pytest.fixture(scope='session')
def aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(scope='session')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session')
def instance_type(request):
    return request.config.getoption('--instance-type')


@pytest.fixture(scope='session')
def accelerator_type(request):
    return request.config.getoption('--accelerator-type')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session')
def tag(request, framework_version, processor):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-py2'.format(framework_version, processor)
    return provided_tag if provided_tag is not None else default_tag


@pytest.fixture(scope='session')
def docker_registry(aws_id, region):
    return '{}.dkr.ecr.{}.amazonaws.com'.format(aws_id, region)


@pytest.fixture(scope='module')
def docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture(scope='module')
def docker_image_uri(docker_registry, docker_image):
    uri = '{}/{}'.format(docker_registry, docker_image)
    return uri


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))
