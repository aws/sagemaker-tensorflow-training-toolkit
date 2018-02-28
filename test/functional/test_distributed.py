import logging

import boto3
import numpy as np
import pytest
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow, TensorFlowModel

from test.resources.python_sdk.timeout import timeout, timeout_and_delete_endpoint

logger = logging.getLogger(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)
logging.getLogger('session.py').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)

script_path = 'test/resources/cifar_10/code'
data_path = 'test/resources/cifar_10/data/training'


@pytest.fixture(scope='module', params=['1.4.1', '1.5.0'])
def tf_version(request):
    return request.param


@pytest.fixture(scope='module', params=['py2'])
def python_version(request):
    return request.param


@pytest.fixture(scope='module', params=['cpu', 'gpu'])
def processor(request):
    return request.param


@pytest.fixture(scope='module')
def docker_image(tf_version, processor, python_version):
    return 'preprod-tensorflow:{}-{}-{}'.format(tf_version, processor, python_version)


@pytest.fixture(scope='module')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


class MyEstimator(TensorFlow):
    def __init__(self, docker_image_uri, **kwargs):
        super(MyEstimator, self).__init__(**kwargs)
        self.docker_image_uri = docker_image_uri

    def train_image(self):
        return self.docker_image_uri

    def create_model(self, model_server_workers=None):
        model = super(MyEstimator, self).create_model()
        model.image = self.docker_image_uri
        return model


def test_distributed(processor, sagemaker_session, docker_image_uri):
    instance_type = 'ml.c4.xlarge' if processor == 'cpu' else 'ml.p2.xlarge'

    with timeout(minutes=15):
        estimator = MyEstimator(entry_point='resnet_cifar_10.py',
                                source_dir=script_path,
                                role='SageMakerRole',
                                training_steps=2,
                                evaluation_steps=1,
                                train_instance_count=2,
                                train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=docker_image_uri)

        logger.info("uploading training data")
        key_prefix = 'integ-test-data/tf-cifar-{}'.format(processor)
        inputs = estimator.sagemaker_session.upload_data(path=data_path,
                                                         key_prefix=key_prefix)

        logger.info("fitting estimator")
        estimator.fit(inputs)

    with timeout_and_delete_endpoint(estimator=estimator, minutes=30):
        logger.info("deploy model")
        json_predictor = estimator.deploy(initial_instance_count=1, instance_type=instance_type)

        data = np.random.rand(32, 32, 3)
        predict_response = json_predictor.predict(data)

        assert len(predict_response['outputs']['probabilities']['floatVal']) == 10
