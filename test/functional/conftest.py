import logging

import boto3
import pytest
from sagemaker import Session

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)


def pytest_addoption(parser):
    parser.addoption('--docker-base-name', action='store', default='preprod-tensorflow')
    parser.addoption('--tag', action='store', default='1.4.1-cpu-py2')
    parser.addoption('--processor', action='store', default='cpu')
    parser.addoption('--region', action='store', default='us-west-2')
    parser.addoption('--aws-id', action='store')


@pytest.fixture(scope='session')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session')
def tag(request):
    return request.config.getoption('--tag')


@pytest.fixture(scope='session')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def aws_id(request):
    return request.config.getoption('--aws-id')


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
