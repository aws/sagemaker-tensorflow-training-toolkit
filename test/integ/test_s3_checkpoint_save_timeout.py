import os
from sagemaker import fw_utils

from test.integ.docker_utils import train
from test.integ.utils import create_config_files, file_exists
import uuid

from test.integ.conftest import SCRIPT_PATH


# https://github.com/tensorflow/tensorflow/issues/15868
def test_s3_checkpoint_save_timeout(docker_image, opt_ml, sagemaker_session):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/python_sdk')

    default_bucket = sagemaker_session.default_bucket()

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                                    bucket=default_bucket,
                                                    s3_key_prefix='test_job',
                                                    script='rand_model_emb.py',
                                                    directory=resource_path)

    checkpoint_s3_path = 's3://{}/integ-s3-timeout/checkpoints-{}'.format(default_bucket,
                                                                          uuid.uuid4())
    hyperparameters = dict(
        training_steps=10,
        evaluation_steps=10,
        checkpoint_path=checkpoint_s3_path
    )
    create_config_files('rand_model_emb.py', s3_source_archive.s3_prefix, opt_ml, hyperparameters)

    train(docker_image, opt_ml)

    assert file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not file_exists(opt_ml, 'output/failure'), 'Failure happened'
