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

## Building your image
Amazon SageMaker utilizes docker containers to run all training & inference jobs.

The docker images are built from the dockerfiles specified in [docker/](https://github.com/aws/sagemaker-tensorflow-containers/tree/master/docker).

The docker files are grouped based on Tensorflow version and separated based on python version and processor type.

The docker images, used to run training & inference jobs, are built from both corresponding "base" and "final" dockerfiles.

### Base Images
> * Tagging scheme is based on <Tensorflow_version>-<processor>-<python_version>. (e.g. 1.4.1-cpu-py2)
> * All "final" dockerfiles build images using base images that use the tagging scheme above.

If you want to build your base docker image, then use:
```
# All build instructions assume you're building from the same directory as the dockerfile.

# CPU
docker build -t tensorflow-base:<Tensorflow_version>-cpu-<python_version> Dockerfile.cpu .

# GPU
docker build -t tensorflow-base:<Tensorflow_version>-gpu-<python_version> Dockerfile.gpu .
```

```
# Example

# CPU
docker build -t tensorflow-base:1.4.1-cpu-py2 Dockerfile.cpu .

# GPU
docker build -t tensorflow-base:1.4.1-gpu-py2 Dockerfile.gpu .
```
### Final Images
> * All "final" dockerfiles use [base images for building](https://github.com/aws/sagemaker-tensorflow-containers/blob/master/docker/1.4.1/final/py2/Dockerfile.cpu#L2).
> * These "base" images are specified with the naming convention of tensorflow-base:<Tensorflow_version>-<processor>-<python_version>.

> * Before building "final" images:
> 1. Build your "base" image. Make sure it is named and tagged in accordance with your "final" dockerfile.
> 2. Create the SageMaker Tensorflow Container python package. (python setup.py sdist)
> 3. Copy your python package to "final" dockerfile directory that you are building. (cp dist/sagemaker_tensorflow_container-*.tar.gz docker/<tensorflow_version>/final/<python_version>)
```
# All build instructions assumes you're building from the same directory as the dockerfile.

# CPU
docker build -t <image_name>:<tag> Dockerfile.cpu .

# GPU
docker build -t <image_name>:<tag> Dockerfile.gpu .
```

```
# Example

# CPU
docker build -t preprod-tensorflow:1.4.1-cpu-py2 Dockerfile.cpu .

# GPU
docker build -t preprod-tensorflow:1.4.1-gpu-py2 Dockerfile.gpu .
```
## Running the tests
Tests are defined in [test/](https://github.com/aws/sagemaker-tensorflow-containers/tree/master/test) and include unit, integration and functional tests.

### Unit Tests
If you want to run unit tests, then use:
```
# All test instructions should be run from the top level directory

pytest test/unit
```

### Integration Tests
> Running integration tests require [docker](https://www.docker.com/) and [AWS credentials](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html), as the integration tests make calls to a couple AWS services.
The integration and functional tests require configurations specified within their respective [conftest.py](https://github.com/aws/sagemaker-tensorflow-containers/blob/master/test/integ/conftest.py).

> Integration tests on GPU require [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

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
