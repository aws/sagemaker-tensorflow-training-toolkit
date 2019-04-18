# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json

import numpy as np


def input_fn(data, content_type):
    """
    Args:
        data: json string containing tensor shape info
              e.g. {"shape": [1, 1024, 1024, 1], "dtype": "float32"}
        content_type: ignored
    Returns:
        a dict with { 'x': numpy array with the specified shape and dtype }
    """

    input_info = json.loads(data)
    shape = input_info['shape']
    dtype = getattr(np, input_info['dtype'])
    return {'x': np.ones(shape, dtype)}
