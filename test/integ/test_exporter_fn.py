import os
import sagemaker
from test.integ.conftest import SCRIPT_PATH


def test_exporter(sdk_estimator_class, local_instance_type, sagemaker_session, run_id):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/iris')
    script_path = os.path.join(resource_path, 'code', 'iris_with_exporter.py')
    data_path = os.path.join(resource_path, 'data')

    s3_dir = sagemaker_session.upload_data(data_path, key_prefix=run_id)
    channels = {'training': '{}/{}'.format(s3_dir, 'training'), 'evaluation': '{}/{}'.format(s3_dir, 'evaluation')}
    print(channels)

    estimator = sdk_estimator_class(entry_point=script_path,
                                    train_instance_type=local_instance_type,
                                    train_instance_count=1,
                                    role='SageMakerRole',
                                    training_steps=10,
                                    evaluation_steps=1)
    estimator.fit(channels)
