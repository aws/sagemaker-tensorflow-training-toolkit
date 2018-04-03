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

import json
import logging
import os
import shutil


def serialize_hyperparameters(hp):
    return {str(k): json.dumps(v) for (k, v) in hp.items()}


def save_as_json(data, filename):
    with open(filename, "wt") as f:
        json.dump(data, f)


def file_exists(resource_folder, file_name):
    return os.path.exists(os.path.join(resource_folder, file_name))


def create_config_files(program, s3_source_archive, path, additional_hp={}):
    rc = {
        "current_host": "algo-1",
        "hosts": ["algo-1"]
    }

    hp = {'sagemaker_region': 'us-west-2',
          'sagemaker_program': program,
          'sagemaker_submit_directory': s3_source_archive,
          'sagemaker_container_log_level': logging.INFO}

    hp.update(additional_hp)

    ic = {
        "training": {"ContentType": "trainingContentType"},
        "evaluation": {"ContentType": "evalContentType"},
        "Validation": {}
    }

    write_conf_files(rc, hp, ic, path)


def write_conf_files(rc, hp, ic, path):
    os.makedirs('{}/input/config'.format(path))

    rc_file = os.path.join(path, 'input/config/resourceconfig.json')
    hp_file = os.path.join(path, 'input/config/hyperparameters.json')
    ic_file = os.path.join(path, 'input/config/inputdataconfig.json')

    hp = serialize_hyperparameters(hp)

    save_as_json(rc, rc_file)
    save_as_json(hp, hp_file)
    save_as_json(ic, ic_file)


def copy_resource(resource_path, opt_ml_path, relative_src_path, relative_dst_path=None):
    if not relative_dst_path:
        relative_dst_path = relative_src_path

    shutil.copytree(os.path.join(resource_path, relative_src_path),
                    os.path.join(opt_ml_path, relative_dst_path))
