# Copyright 2017-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.tensorflow import TensorFlow
from sagemaker.utils import unique_name_from_base


RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


def test_multi_node(
    sagemaker_session, instance_type, image_uri, tmpdir, framework_version
):
    estimator = TensorFlow(
        entry_point=os.path.join(
            RESOURCE_PATH, "multi_worker_mirrored", "train_sample.py"
        ),
        role="SageMakerRole",
        instance_type=instance_type,
        instance_count=2,
        image_name=image_uri,
        framework_version=framework_version,
        py_version="py3",
        hyperparameters={
            "sagemaker_multi_worker_mirrored_enabled": True,
        },
        sagemaker_session=sagemaker_session,
    )
    estimator.fit(job_name=unique_name_from_base("test-tf-mwms"))
    raise NotImplementedError("Yet to add assertion")
