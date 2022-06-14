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

import pytest

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


def test_keras_example(
    sagemaker_session, instance_type, image_uri, tmpdir, framework_version, capsys
):
    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, "multi_worker_mirrored", "train_dummy.py"),
        role="SageMakerRole",
        instance_type=instance_type,
        instance_count=2,
        image_name=image_uri,
        framework_version=framework_version,
        py_version="py3",
        hyperparameters={
            "sagemaker_multi_worker_mirrored_strategy_enabled": True,
        },
        sagemaker_session=sagemaker_session,
    )
    estimator.fit(job_name=unique_name_from_base("test-tf-mwms"))
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    assert "Running distributed training job with multi_worker_mirrored_strategy setup" in logs
    assert "TF_CONFIG=" in logs


@pytest.mark.skip_cpu
def test_tf_model_garden(
    sagemaker_session, instance_type, image_uri, tmpdir, framework_version, capsys
):
    epochs = 10
    global_batch_size = 64
    train_steps = int(1024 * epochs / global_batch_size)
    steps_per_loop = train_steps // 10
    overrides = (
        f"runtime.enable_xla=False,"
        f"runtime.num_gpus=1,"
        f"runtime.distribution_strategy=multi_worker_mirrored,"
        f"runtime.mixed_precision_dtype=float16,"
        f"task.train_data.global_batch_size={global_batch_size},"
        f"task.train_data.input_path=/opt/ml/input/data/training/validation*,"
        f"task.train_data.cache=True,"
        f"trainer.train_steps={train_steps},"
        f"trainer.steps_per_loop={steps_per_loop},"
        f"trainer.summary_interval={steps_per_loop},"
        f"trainer.checkpoint_interval={train_steps},"
        f"task.model.backbone.type=resnet,"
        f"task.model.backbone.resnet.model_id=50"
    )
    estimator = TensorFlow(
        git_config={
            "repo": "https://github.com/tensorflow/models.git",
            "branch": "v2.9.2",
        },
        source_dir=".",
        entry_point="official/vision/train.py",
        model_dir=False,
        instance_type=instance_type,
        instance_count=2,
        image_uri=image_uri,
        hyperparameters={
            "sagemaker_multi_worker_mirrored_strategy_enabled": True,
            "experiment": "resnet_imagenet",
            "config_file": "official/vision/configs/experiments/image_classification/imagenet_resnet50_gpu.yaml",
            "mode": "train",
            "model_dir": "/opt/ml/model",
            "params_override": overrides,
        },
        max_run=60 * 60 * 1,  # 1 hour
        role="SageMakerRole",
    )
    estimator.fit(
        inputs="s3://collection-of-ml-datasets/Imagenet/TFRecords/validation",
        job_name=unique_name_from_base("test-tf-mwms"),
    )
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    assert "Running distributed training job with multi_worker_mirrored_strategy setup" in logs
    assert "TF_CONFIG=" in logs
