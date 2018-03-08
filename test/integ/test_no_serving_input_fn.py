import os

from sagemaker import fw_utils

from test.integ.docker_utils import train
from test.integ.utils import copy_resource, create_config_files, file_exists
from test.integ.conftest import SCRIPT_PATH


def test_no_serving_input_fn(docker_image, sagemaker_session, opt_ml):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/iris')

    copy_resource(resource_path, opt_ml, 'code')
    copy_resource(resource_path, opt_ml, 'data', 'input/data')

    s3_source_archive = fw_utils.tar_and_upload_dir(session=sagemaker_session.boto_session,
                                bucket=sagemaker_session.default_bucket(),
                                s3_key_prefix='test_job',
                                script='iris_no_serving_input.py',
                                directory=os.path.join(resource_path, 'code'))

    additional_hyperparameters = dict(training_steps=1, evaluation_steps=1)
    create_config_files('iris_no_serving_input.py', s3_source_archive.s3_prefix, opt_ml, additional_hyperparameters)

    os.makedirs(os.path.join(opt_ml, 'model'))

    train(docker_image, opt_ml)

    # model is not saved so do not expect it there
    assert not file_exists(opt_ml, 'model/export/Servo'), 'model was not exported'
    assert file_exists(opt_ml, 'model/checkpoint'), 'checkpoint was not created'
    assert file_exists(opt_ml, 'output/success'), 'Success file was not created'
    assert not file_exists(opt_ml, 'output/failure'), 'Failure happened'
