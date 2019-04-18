#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os

import tensorflow as tf

from test.integ.docker_utils import HostingContainer
from test.integ.utils import copy_resource
from test.integ.conftest import SCRIPT_PATH


def create_model(export_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session() as session:
        x = tf.placeholder(tf.float32, shape=[None, 1024, 1024, 1], name='x')
        a = tf.constant(2.0)

        y = tf.multiply(a, x, name='y')
        predict_signature_def = (
            tf.saved_model.signature_def_utils.predict_signature_def({
                'x': x
            }, {'y': y}))
        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature_def
        }
        session.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)
        builder.save()


def test_large_grpc_message(docker_image, opt_ml, processor, region):
    resource_path = os.path.join(SCRIPT_PATH, '../resources/large_grpc_message')
    copy_resource(resource_path, opt_ml, 'code', 'code')
    export_dir = os.path.join(opt_ml, 'model', 'export', 'Servo', '1')
    create_model(export_dir)

    with HostingContainer(opt_ml=opt_ml, image=docker_image,
                          script_name='inference.py',
                          processor=processor,
                          region=region) as c:
        c.execute_pytest('test/integ/container_tests/large_grpc_message.py')
