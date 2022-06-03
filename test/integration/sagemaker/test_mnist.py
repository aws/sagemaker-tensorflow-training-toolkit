# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import boto3
import pytest
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.utils import unique_name_from_base
from six.moves.urllib.parse import urlparse

from timeout import timeout


@pytest.mark.deploy_test
def test_mnist(sagemaker_session, image_uri, instance_type, framework_version):
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    script = os.path.join(resource_path, "mnist", "mnist.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        framework_version=framework_version,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, "mnist", "data"), key_prefix="scriptmode/mnist"
    )
    estimator.fit(inputs, job_name=unique_name_from_base("test-sagemaker-mnist"))
    _assert_s3_file_exists(sagemaker_session.boto_region_name, estimator.model_data)


def test_distributed_mnist_no_ps(sagemaker_session, image_uri, instance_type, framework_version):
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    script = os.path.join(resource_path, "mnist", "mnist.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_count=2,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        framework_version=framework_version,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, "mnist", "data"), key_prefix="scriptmode/mnist"
    )
    estimator.fit(inputs, job_name=unique_name_from_base("test-tf-sm-distributed-mnist"))
    _assert_s3_file_exists(sagemaker_session.boto_region_name, estimator.model_data)


def test_distributed_mnist_ps(sagemaker_session, image_uri, instance_type, framework_version):
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    script = os.path.join(resource_path, "mnist", "mnist_custom.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        hyperparameters={"sagemaker_parameter_server_enabled": True},
        instance_count=2,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        framework_version=framework_version,
    )
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, "mnist", "data-distributed"),
        key_prefix="scriptmode/mnist-distributed",
    )
    estimator.fit(inputs, job_name=unique_name_from_base("test-tf-sm-distributed-mnist"))
    _assert_checkpoint_exists_v2(sagemaker_session.boto_region_name, estimator.model_dir, 10)


def test_tuning(sagemaker_session, image_uri, instance_type, framework_version):
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    script = os.path.join(resource_path, "mnist", "mnist.py")

    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        image_uri=image_uri,
        framework_version=framework_version,
        script_mode=True,
    )

    hyperparameter_ranges = {"epochs": IntegerParameter(1, 2)}
    objective_metric_name = "accuracy"
    metric_definitions = [{"Name": objective_metric_name, "Regex": "accuracy = ([0-9\\.]+)"}]

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=2,
        max_parallel_jobs=2,
    )

    with timeout(minutes=20):
        inputs = estimator.sagemaker_session.upload_data(
            path=os.path.join(resource_path, "mnist", "data"), key_prefix="scriptmode/mnist"
        )

        tuning_job_name = unique_name_from_base("test-tf-sm-tuning", max_length=32)
        tuner.fit(inputs, job_name=tuning_job_name)
        tuner.wait()


def _assert_checkpoint_exists_v2(region, model_dir, checkpoint_number):
    """
    Checking for v2 style checkpoints i.e. checkpoint and .index files
    """
    _assert_s3_file_exists(region, os.path.join(model_dir, 'checkpoint'))
    _assert_s3_file_exists(region,
                           os.path.join(model_dir, 'model.ckpt-{}.index'.format(checkpoint_number)))


def _assert_checkpoint_exists(region, model_dir, checkpoint_number):
    _assert_s3_file_exists(region, os.path.join(model_dir, "graph.pbtxt"))
    _assert_s3_file_exists(
        region, os.path.join(model_dir, "model.ckpt-{}.index".format(checkpoint_number))
    )
    _assert_s3_file_exists(
        region, os.path.join(model_dir, "model.ckpt-{}.meta".format(checkpoint_number))
    )


def _assert_s3_file_exists(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource("s3", region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip("/")).load()
