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
import tarfile

import pytest
from sagemaker.tensorflow import TensorFlow

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
TF_CHECKPOINT_FILES = ["graph.pbtxt", "model.ckpt-0.index", "model.ckpt-0.meta"]


@pytest.fixture  # noqa: F811
def py_full_version(py_version):  # noqa: F811
    if py_version == "2":
        return "2.7"
    else:
        return "3.6"


@pytest.mark.skip_gpu
def test_mnist_cpu(sagemaker_local_session, image_uri, tmpdir, framework_version):
    output_path = "file://{}".format(tmpdir)
    run_tf_training(
        script=os.path.join(RESOURCE_PATH, "mnist", "mnist.py"),
        instance_type="local",
        instance_count=1,
        sagemaker_local_session=sagemaker_local_session,
        image_uri=image_uri,
        framework_version=framework_version,
        output_path=output_path,
        training_data_path="file://{}".format(os.path.join(RESOURCE_PATH, "mnist", "data")),
    )
    _assert_files_exist_in_tar(output_path, ["my_model.h5"])


@pytest.mark.skip_gpu
def test_distributed_training_cpu_no_ps(
    sagemaker_local_session, image_uri, tmpdir, framework_version
):
    output_path = "file://{}".format(tmpdir)
    run_tf_training(
        script=os.path.join(RESOURCE_PATH, "mnist", "mnist_estimator.py"),
        instance_type="local",
        instance_count=2,
        sagemaker_local_session=sagemaker_local_session,
        image_uri=image_uri,
        framework_version=framework_version,
        output_path=output_path,
        training_data_path="file://{}".format(
            os.path.join(RESOURCE_PATH, "mnist", "data-distributed")
        ),
    )
    _assert_files_exist_in_tar(output_path, TF_CHECKPOINT_FILES)


@pytest.mark.skip_gpu
def test_distributed_training_cpu_ps(sagemaker_local_session, image_uri, tmpdir, framework_version):
    output_path = "file://{}".format(tmpdir)
    run_tf_training(
        script=os.path.join(RESOURCE_PATH, "mnist", "mnist_estimator.py"),
        instance_type="local",
        instance_count=2,
        sagemaker_local_session=sagemaker_local_session,
        image_uri=image_uri,
        framework_version=framework_version,
        output_path=output_path,
        hyperparameters={"sagemaker_parameter_server_enabled": True},
        training_data_path="file://{}".format(
            os.path.join(RESOURCE_PATH, "mnist", "data-distributed")
        ),
    )
    _assert_files_exist_in_tar(output_path, TF_CHECKPOINT_FILES)


def run_tf_training(
    script,
    instance_type,
    instance_count,
    sagemaker_local_session,
    image_uri,
    framework_version,
    training_data_path,
    output_path=None,
    hyperparameters=None,
):

    hyperparameters = hyperparameters or {}

    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_local_session,
        image_name=image_uri,
        model_dir="/opt/ml/model",
        output_path=output_path,
        hyperparameters=hyperparameters,
        base_job_name="test-tf",
        framework_version=framework_version,
        py_version="py3",
    )

    estimator.fit(training_data_path)


def _assert_files_exist_in_tar(output_path, files):
    if output_path.startswith("file://"):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, "model.tar.gz")
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)
