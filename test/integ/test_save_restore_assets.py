import os

from sagemaker import fw_utils

from test.integ.docker_utils import train, HostingContainer
from test.integ.utils import copy_resource, create_config_files
from test.integ.conftest import SCRIPT_PATH

import uuid


def test_save_restore_assets(docker_image, sagemaker_session, opt_ml, processor):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/sentiment')

    default_bucket = sagemaker_session.default_bucket()

    copy_resource(resource_path, opt_ml, 'data', 'input/data')

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                bucket=sagemaker_session.default_bucket(),
                                s3_key_prefix='test_job',
                                script='sentiment.py',
                                directory=os.path.join(resource_path, 'code'))

    checkpoint_s3_path = 's3://{}/save_restore_assets/output-{}'.format(default_bucket, uuid.uuid4())

    additional_hyperparameters = dict(
        training_steps=1000,
        evaluation_steps=100,
        checkpoint_path=checkpoint_s3_path)
    create_config_files('sentiment.py', s3_source_archive.s3_prefix, opt_ml, additional_hyperparameters)
    os.makedirs(os.path.join(opt_ml, 'model'))

    train(docker_image, opt_ml, processor)

    with HostingContainer(opt_ml=opt_ml, image=docker_image, script_name='sentiment.py', processor=processor) as c:
        c.execute_pytest('test/integ/container_tests/sentiment_classification.py')
