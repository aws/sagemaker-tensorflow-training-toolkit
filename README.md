## SageMaker TensorFlow Containers

SageMaker Tensorflow Containers is an open source library for making the TensorFlow framework run on Amazon SageMaker.

For information on running Tensorflow jobs on SageMaker: [Python SDK](https://github.com/aws/sagemaker-python-sdk).

For notebook examples: [SageMaker Notebook Examples](https://github.com/awslabs/amazon-sagemaker-examples).

## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your development machine:
* [Docker](https://www.docker.com/)

#### For Testing on GPU
* [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)

#### Recommended
* A python environment management tool. (e.g. [PyEnv](https://github.com/pyenv/pyenv), [VirtualEnv](https://virtualenv.pypa.io/en/stable/))

## Running the tests
Tests are defined in test/ and include unit, integration and functional tests.

### Unit Tests
If you want to run unit tests, then use:
```
# All test instructions should be run from the top level directory

pytest test/unit
```

### Integration Tests
> Running integration tests require [docker](https://www.docker.com/) and [AWS credentials](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html), as the integration tests make calls to a couple AWS services.
The integration and functional tests require configurations specified within their respective [conftest.py](https://github.com/aws/sagemaker-tensorflow-containers/blob/master/test/integ/conftest.py).

> Before running integration tests:
> 1. Build your docker image.
> 2. Pass in the correct pytest arguments to run tests against your docker image.

If you want to run local integration tests, then use:
```
# Required arguments for integration tests are found in test/integ/conftest.py

pytest test/integ --docker-base-name <your_docker_image> \
                   --tag <your_docker_image_tag> \
                   --framework-version <tensorflow_version> \
                   --processor <cpu_or_gpu>
```

```
# Example
pytest test/integ --docker-base-name preprod-tensorflow \
                   --tag 1.0 \
                   --framework-version 1.4.1 \
                   --processor cpu
```

### Functional Tests
If you want to run a functional end to end test on [Amazon SageMaker](https://aws.amazon.com/sagemaker/), then use:

> * Functional tests require your docker image to be within an [Amazon ECR repository]().
> * The docker-base-name is your [ECR repository namespace](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Repositories.html).
> * The instance-type is your specified [Amazon SageMaker Instance Type](https://aws.amazon.com/sagemaker/pricing/instance-types/) that the functional test will run on.

> Before running functional tests:
> 1. Build your docker image.
> 2. Push the image to your ECR repository.
> 3. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

```
# Required arguments for integration tests are found in test/functional/conftest.py

pytest test/functional --aws-id <your_aws_id> \
                   --docker-base-name <your_docker_image> \
                   --instance-type <amazon_sagemaker_instance_type> \
                   --tag <your_docker_image_tag> \
```

```
# Example
pytest test/functional --aws-id 12345678910 \
                   --docker-base-name preprod-tensorflow \
                   --instance-type ml.m4.xlarge \
                   --tag 1.0
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-tensorflow-containers/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License.
