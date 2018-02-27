import logging
import os
import platform

import pytest
import shutil
import tempfile

import test.integ.docker_utils as d

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption('--stage', action='store', default='dev')
    parser.addoption('--region', action='store', default='us-west-2')
    parser.addoption('--tag', action='store', default='1.0')
    parser.addoption('--aws-id', action='store')


@pytest.fixture(scope='session')
def stage(request):
    return request.config.getoption('--stage')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def tag(request):
    return request.config.getoption('--tag')


@pytest.fixture(scope='session')
def aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(scope='session')
def docker_registry(aws_id, region):
    d.login(aws_id, region)
    return '{}.dkr.ecr.{}.amazonaws.com'.format(aws_id, region)


@pytest.fixture(scope='module')
def docker_image(tag):
    return 'sagemaker-tensorflow-py2-cpu:{}'.format(tag)


@pytest.fixture(scope='module')
def docker_image_uri(docker_registry, docker_image):
    uri = '{}/{}'.format(docker_registry, docker_image)
    return uri


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)
