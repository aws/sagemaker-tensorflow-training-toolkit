import pytest
from test.integ.docker_utils import Container


@pytest.fixture
def required_versions(tf_version):
    if tf_version == '1.4.1':
        return ['tensorflow-serving-api==1.4.0',
                'tensorflow-tensorboard==0.4.0',
                'tensorflow==1.4.1']
    elif tf_version == '1.5.0':
        return ['tensorflow-serving-api==1.5.0',
                'tensorflow-tensorboard==1.5.1',
                'tensorflow==1.5.0']
    else:
        raise ValueError("invalid internal test config")


def test_framework_versions(docker_image, required_versions):
    with Container(docker_image) as c:
        output = c.execute_command(['pip', 'freeze'])
        lines = output.splitlines()
        result = sorted([v for v in lines if v in required_versions])

        assert required_versions == result
