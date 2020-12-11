# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.utils import unique_name_from_base

from integration import DEFAULT_TIMEOUT, RESOURCE_PATH
from integration.sagemaker.timeout import timeout


@pytest.mark.skip_cpu
@pytest.mark.skip_generic
@pytest.mark.parametrize(
    "instances, instance_type",
    [(2, "ml.p3.16xlarge")],
)
def test_smdataparallel_training(instances, instance_type, sagemaker_session, image_uri, framework_version, tmpdir):
    default_bucket = sagemaker_session.default_bucket()
    output_path = "s3://" + os.path.join(default_bucket, "tensorflow/smdataparallel")

    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, "mnist", "smdataparallel_mnist.py"),
        role="SageMakerRole",
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        instance_count=instances,
        image_uri=image_uri,
        output_path=output_path,
        framework_version=framework_version,
        py_version="py3",
        distribution={"smdistributed": {"dataparallel": {"enabled": True}}}
    )

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator.fit(job_name=unique_name_from_base("test-tf-smdataparallel"))

        model_data_source = sagemaker.local.data.get_data_source_instance(
            estimator.model_data, sagemaker_session
        )

        for filename in model_data_source.get_file_list():
            assert os.path.basename(filename) == "model.tar.gz"
