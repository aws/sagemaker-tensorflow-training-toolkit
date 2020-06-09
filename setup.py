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

from glob import glob
import os
from os.path import basename
from os.path import splitext
import sys

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read('VERSION').strip()


test_dependencies = ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock',
                     'sagemaker==1.50.1', 'tensorflow<2.0', 'docker-compose', 'boto3==1.10.50',
                     'six==1.13.0', 'python-dateutil>=2.1,<2.8.1', 'botocore==1.13.50',
                     'requests-mock', 'awscli==1.16.314']

if sys.version_info.major > 2:
    test_dependencies.append('sagemaker-experiments==0.1.7')

setup(
    name='sagemaker_tensorflow_training',
    version=read_version(),
    description='Open source library for creating '
                'TensorFlow containers to run on Amazon SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-tensorflow-containers',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    install_requires=['sagemaker-training>=3.5.2', 'numpy', 'scipy', 'sklearn',
                      'pandas', 'Pillow', 'h5py'],
    extras_require={
        'test': test_dependencies,
        'benchmark': ['click']
    },
)
