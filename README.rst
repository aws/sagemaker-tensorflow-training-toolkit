=====================================
SageMaker TensorFlow Training Toolkit
=====================================

The SageMaker TensorFlow Training Toolkit is an open source library for making the
TensorFlow framework run on `Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__.

This repository also contains Dockerfiles which install this library, TensorFlow, and dependencies
for building SageMaker TensorFlow images.

For information on running TensorFlow jobs on SageMaker:

- `SageMaker Python SDK documentation <https://sagemaker.readthedocs.io/en/stable/using_tf.html>`__
- `SageMaker Notebook Examples <https://github.com/awslabs/amazon-sagemaker-examples>`__

Table of Contents
-----------------

#. `Getting Started <#getting-started>`__
#. `Building your Image <#building-your-image>`__
#. `Running the tests <#running-the-tests>`__

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your
development machine:

- `Docker <https://www.docker.com/>`__

For Testing on GPU
^^^^^^^^^^^^^^^^^^

-  `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__

Recommended
^^^^^^^^^^^

-  A Python environment management tool. (e.g.
   `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building your Image
-------------------

`Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__
utilizes Docker containers to run all training jobs & inference endpoints.

The Docker images are built from the Dockerfiles specified in
`docker/ <https://github.com/aws/sagemaker-tensorflow-containers/tree/master/docker>`__.

The Dockerfiles are grouped based on TensorFlow version and separated
based on Python version and processor type.

The Dockerfiles for TensorFlow 2.0+ are available in the
`tf-2 <https://github.com/aws/sagemaker-tensorflow-container/tree/tf-2>`__ branch.

To build the images, first copy the files under
`docker/build_artifacts/ <https://github.com/aws/sagemaker-tensorflow-container/tree/tf-2/docker/build_artifacts>`__
to the folder container the Dockerfile you wish to build.

::

    # Example for building a TF 2.1 image with Python 3
    cp docker/build_artifacts/* docker/2.1.0/py3/.

After that, go to the directory containing the Dockerfile you wish to build,
and run ``docker build`` to build the image.

::

    # Example for building a TF 2.1 image for CPU with Python 3
    cd docker/2.1.0/py3
    docker build -t tensorflow-training:2.1.0-cpu-py3 -f Dockerfile.cpu .

Don't forget the period at the end of the ``docker build`` command!

Running the tests
-----------------

Running the tests requires installation of the SageMaker TensorFlow Training Toolkit code and its test
dependencies.

::

    git clone https://github.com/aws/sagemaker-tensorflow-container.git
    cd sagemaker-tensorflow-container
    pip install -e .[test]

Tests are defined in
`test/ <https://github.com/aws/sagemaker-tensorflow-container/tree/master/test>`__
and include unit, integration and functional tests.

Unit Tests
~~~~~~~~~~

If you want to run unit tests, then use:

::

    # All test instructions should be run from the top level directory
    pytest test/unit

Integration Tests
~~~~~~~~~~~~~~~~~

Running integration tests require `Docker <https://www.docker.com/>`__ and `AWS
credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the integration tests make calls to a couple AWS services. The integration and functional
tests require configurations specified within their respective
`conftest.py <https://github.com/aws/sagemaker-tensorflow-containers/blob/master/test/integration/conftest.py>`__.Make sure to update the account-id and region at a minimum.

Integration tests on GPU require `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__.

Before running integration tests:

#. Build your Docker image.
#. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

::

    # Required arguments for integration tests are found in test/integ/conftest.py
    pytest test/integration --docker-base-name <your_docker_image> \
                            --tag <your_docker_image_tag> \
                            --framework-version <tensorflow_version> \
                            --processor <cpu_or_gpu>

::

    # Example
    pytest test/integration --docker-base-name preprod-tensorflow \
                            --tag 1.0 \
                            --framework-version 1.4.1 \
                            --processor cpu

Functional Tests
~~~~~~~~~~~~~~~~

Functional tests are removed from the current branch, please see them in older branch `r1.0 <https://github.com/aws/sagemaker-tensorflow-container/tree/r1.0#functional-tests>`__.

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-tensorflow-containers/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

License
-------

SageMaker TensorFlow Containers is licensed under the Apache 2.0 License. It is copyright 2018
Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/
