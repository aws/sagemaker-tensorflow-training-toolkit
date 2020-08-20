# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Benchmark script for TensorFlow.

Originally copied from:
https://github.com/tensorflow/benchmarks/blob/e3bd1370ba21b02c4d34340934ffb4941977d96f/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py
"""
from __future__ import absolute_import, division, print_function

from absl import app
from absl import flags as absl_flags
import tensorflow.compat.v1 as tf

import benchmark_cnn
import cnn_util
import flags
import mlperf
from cnn_util import log_fn


flags.define_flags()
for name in flags.param_specs.keys():
    absl_flags.declare_key_flag(name)

absl_flags.DEFINE_boolean(
    "ml_perf_compliance_logging",
    False,
    "Print logs required to be compliant with MLPerf. If set, must clone the "
    "MLPerf training repo https://github.com/mlperf/training and add "
    "https://github.com/mlperf/training/tree/master/compliance to the "
    "PYTHONPATH",
)


def main(positional_arguments):
    # Command-line arguments like '--distortions False' are equivalent to
    # '--distortions=True False', where False is a positional argument. To prevent
    # this from silently running with distortions, we do not allow positional
    # arguments.
    assert len(positional_arguments) >= 1
    if len(positional_arguments) > 1:
        raise ValueError("Received unknown positional arguments: %s" % positional_arguments[1:])

    params = benchmark_cnn.make_params_from_flags()
    with mlperf.mlperf_logger(absl_flags.FLAGS.ml_perf_compliance_logging, params.model):
        params = benchmark_cnn.setup(params)
        bench = benchmark_cnn.BenchmarkCNN(params)

    tfversion = cnn_util.tensorflow_version_tuple()
    log_fn("TensorFlow:  %i.%i" % (tfversion[0], tfversion[1]))

    bench.print_info()
    bench.run()


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
