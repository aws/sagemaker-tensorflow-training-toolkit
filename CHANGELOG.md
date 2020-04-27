# Changelog

## v2.3.2 (2020-04-07)

### Bug Fixes and Other Changes

 * Bump smdebug version

## v2.3.1 (2020-04-06)

### Bug Fixes and Other Changes

 * updating pillow version of tf1.15

## v2.3.0 (2020-04-02)

### Features

 * install sagemaker-tensorflow-toolkit from PyPI.

## v2.2.8 (2020-04-01)

### Bug Fixes and Other Changes

 * Allowing arguments for deep_learning_container.py for tf1.15

## v2.2.7.post0 (2020-03-31)

### Testing and Release Infrastructure

 * refactor toolkit tests.

## v2.2.7 (2020-03-26)

### Bug Fixes and Other Changes

 * Adding of deep_learning_container.py in Tf1.15

## v2.2.6 (2020-03-16)

### Bug Fixes and Other Changes

 * smdebug 0.7.1
 * Added marker to skip on pipeline

## v2.2.5 (2020-03-12)

### Bug Fixes and Other Changes

 * install smexperiments when python >= 3.6
 * SM integration test for TF 1.x
 * upgrade to latest sagemaker-experiments
 * Added pytest fixture

## v2.2.4 (2020-03-11)

### Bug Fixes and Other Changes

 * update smdebug wheel
 * Revert "Update smdebug to 0.7.0 - TF 1.15.2 (#298)"

## v2.2.3 (2020-03-10)

### Bug Fixes and Other Changes

 * update smdebug wheel
 * Update smdebug to 0.7.0 - TF 1.15.2
 * install SageMaker Python SDK into Python 3 images

## v2.2.2.post0 (2020-03-05)

### Testing and Release Infrastructure

 * fix PR build

## v2.2.2 (2020-02-20)

### Bug Fixes and Other Changes

 * copy all tests to test-toolkit folder.

## v2.2.1 (2020-02-17)

### Bug Fixes and Other Changes

 * update: update r1.15.2 dockerfiles

## v2.2.0 (2020-02-13)

### Features

 * Add release to PyPI. Change package name to sagemaker-tensorflow-training.

### Bug Fixes and Other Changes

 * pin awscli to latest version
 * Pin awscli to latest
 * bump smdebug version to 0.5.0.post0
 * update: Update awscli version and remove related pins
 * update: Update buildspec for TF 1.15.0
 * update copyright year in license header

### Documentation Changes

 * update README.rst
 * Add link to TF 2.0 branch

### Testing and Release Infrastructure

 * Add twine check during PR.
 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v0.1.0 (2020-02-12)

### Features

 * Add release to PyPI. Change package name to sagemaker-tensorflow-training.

### Bug Fixes and Other Changes

 * pin awscli to latest version
 * Pin awscli to latest
 * bump smdebug version to 0.5.0.post0
 * update: Update awscli version and remove related pins
 * update: Update buildspec for TF 1.15.0
 * update copyright year in license header
 * update: Release TF 1.15.0 dockerfiles
 * use regional endpoint for STS in builds
 * update documentation link in warning message
 * update instance type region availability.
 * license file was missing from root of repo.
 * install tensorflow<2.0
 * merge dockerfiles
 * move script mode branch to master
 * use last legacy mode version for --framework-version test arg default
 * Pin pytest and pluggy to work around configparser error
 * Use multiprocessing.Process to launch parameter server
 * increase grpc message size limit to 2gb
 * Fix typo in serving method name
 * restore python-dev package in image
 * Add default tag to functional tests
 * update link to correct docs
 * Add EI Dockerfile for 1.11
 * Add EI documentation within README
 * add Dockerfile for EI
 * Use get_closest_marker instead of get_marker
 * Add docker files of TF 1.12
 * Default GRPC timeout for EI & Allow timeout to be configurable
 * remove requests from test dependencies
 * catch RpcError due to change in GRPC
 * Icywang86rui gpu fix
 * Read port range from container support for TFS port
 * Unfreeze requests version
 * export_saved_model: copy asset files
 * add port to dockerfile
 * Updated TF Pipe Mode Version
 * Fix MKL setting
 * Set MKL vars plus tests
 * increase test timeout
 * Add back https to S3
 * Add 1.11.0 CPU and GPU Dockerfile
 * pin requests version
 * fix memory leak in serving
 * Update region in s3 boto client in serve
 * Update readme with instructions for 1.9.0 and above
 * Fix deserialization of dicts for json predict requests
 * Add dockerfile and update test for tensorflow 1.10.0
 * Support tensorflow 1.9.0
 * Add integ tests to verify that tensorflow in gpu-image can access gpu-devices.
 * train on 3 epochs for pipe mode test
 * Change error classes used by _default_input_fn() and _default_output_fn()
 * Changing assertion to check only existence
 * Install sagemaker-tensorflow from pypi. Add MKL environment variables for TF 1.8
 * get most recent saved model to export
 * pip install tensorflow 1.8 in 1.8 cpu image
 * install tensorflow extensions
 * upgrade cpu binaries in docker build
 * Force upgrade of the framework binaries to make sure the right binaries are installed.
 * Add Pillow to pip install list
 * Increase train steps for cifar distributed test to mitigate race condition
 * Add TensorFlow 1.8 dockerfiles
 * Add TensorFlow 1.7 dockerfiles
 * Explain how to download tf binaries from PyPI
 * Allow training without S3
 * Fix hyperparameter name for detecting a tuning job
 * Checkout v1.4.1 tag instead of r1.4 branch
 * Move processing of requirements file in.
 * Generate checkpoint path using TRAINING_JOB_NAME environment variable if needed
 * Wrap user-provided model_fn to pass arguments positionally (maintains compatibility with existing behavior)
 * Add more unit tests for trainer, fix __all__ and rename train.py to avoid import conflict
 * Use regional endpoint for S3 client
 * Update README.rst
 * Pass input_channels to eval_input_fn if defined
 * Fix setup.py to refer to renamed README
 * Add test and build instructions
 * Fix year in license headers
 * Add TensorFlow 1.6
 * Add test instructions in README
 * Add container support to install_requires
 * Add Apache license headers
 * Use wget to install tensorflow-model-server
 * Fix file path for integ test
 * Fix s3_prefix path in integ test
 * Fix typo in path for integ test
 * Add input_channels to train_input_fn interface.
 * Update logging and make serving_input_fn optional.
 * remove pip install in tensorflow training
 * Modify integration tests to run nvidia-docker for gpu
 * add h5py for keras models
 * Add local integ tests & resources
 * Restructure repo to use a directory per TF version for dockerfiles
 * Rename "feature_map" variables to "feature_dict" to avoid overloading it with the ML term "feature map"
 * Copying in changes from internal repo:
 * Add functional test
 * Fix FROM image names for final build dockerfiles
 * Add dockerfiles for building our production images (TF 1.4)
 * GPU Dockerfile and setup.py fixes
 * Add base image Dockerfiles for 1.4
 * Merge pull request #1 from aws/mvs-first-commit
 * first commit
 * Updating initial README.md from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Initial commit

### Documentation Changes

 * update README.rst
 * Add link to TF 2.0 branch

### Testing and Release Infrastructure

 * Add twine check during PR.
 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v0.1.0 (2019-05-22)

### Bug fixes and other changes

## v2.0.7 (2019-08-15)

### Bug fixes and other changes

 * update no-p2 and no-p3 regions.

## v2.0.6 (2019-08-01)

### Bug fixes and other changes

 * fix horovod mnist script

## v2.0.5 (2019-06-17)

### Bug fixes and other changes

 * bump sagemaker-containers version to 2.4.10
 * add hyperparameter tuning test

## v2.0.4 (2019-06-06)

### Bug fixes and other changes

 * fix integ test errors when running with py2

## v2.0.3 (2019-06-06)

### Bug fixes and other changes

 * only run one test during deployment

## v2.0.2 (2019-06-04)

### Bug fixes and other changes

 * resolve pluggy version conflict

## v2.0.1 (2019-06-03)

### Bug fixes and other changes

 * remove non-ascii character in CHANGELOG
 * remove extra comma in buildspec-release.yml

## v2.0.0 (2019-06-03)

### Bug fixes and other changes

 * Parameterize processor and py_version for test runs
 * use unique name for integration job hyperparameter tuning job
 * fix flake8 errors and add flake8 run in buildspec.yml
 * skip gpu SageMaker test in regions with limited amount of p2/3 instances
 * skip setup on second remote run
 * add setup file back
 * add branch name to remote gpu test run command
 * remove setup file in release build gpu test
 * ignore coverage in release build tests
 * use tar file name as framework_support_installable in build_all.py
 * Add release build
 * Explicitly set lower-bound for botocore version
 * Pull request to test codebuild trigger on TensorFlow script mode
 * Update integ test for checking Python version
 * Upgrade to TensorFlow 1.13.1
 * Add mpi4py to pip installs
 * Add SageMaker integ test for hyperparameter tuning model_dir logic
 * Add Horovod benchmark
 * Fix model_dir adjustment for hyperparameter tuning jobs
 * change model_dir to training job name if it is for tuning.
 * Tune test_s3_plugin test
 * Skip the s3_plugin test before new binary released
 * Add model saving warning at end of training
 * Specify region when creating S3 resource in integ tests
 * Fix instance_type fixture setup for tests
 * Read framework version from Python SDK for integ test default
 * Fix SageMaker Session handling in Horovod test
 * Configure encoding to be utf-8
 * Use the test argement framework_version in all tests
 * Fix broken test test_distributed_mnist_no_ps
 * Add S3 plugin tests
 * Skip horovod local CPU test in GPU instances
 * Add Horovod tests
 * Skip horovod integration tests
 * TensorFlow 1.12 and Horovod support
 * Deprecate get_marker. Use get_closest_marker instead
 * Force parameter server to run on CPU
 * Add python-dev and build-essential to Dockerfiles
 * Update script_mode_train_any_tf_script_in_sage_maker.ipynb
 * Skip keras local mode test on gpu and use random port for serving in the test
 * Fix Keras test
 * Create parameter server in different thread
 * Add Keras support
 * Fix broken unit tests
 * Unset CUDA_VISIBLE_DEVICES for worker processes
 * Disable GPU for parameter process
 * Set parameter process waiting to False
 * Update sagemaker containers
 * GPU fix
 * Set S3 environment variables
 * Add CI configuration files
 * Add distributed training support
 * Edited the tf script mode notebook
 * Add benchmarking script
 * Add Script Mode example
 * Add integration tests to run training jobs with sagemaker
 * Add tox.ini and configure coverage and flake runs
 * Scriptmode single machine training implementation
 * Update region in s3 boto client in serve
 * Update readme with instructions for 1.9.0 and above
 * Fix deserialization of dicts for json predict requests
 * Add dockerfile and update test for tensorflow 1.10.0
 * Support tensorflow 1.9.0
 * Add integ tests to verify that tensorflow in gpu-image can access gpu-devices.
 * train on 3 epochs for pipe mode test
 * Change error classes used by _default_input_fn() and _default_output_fn()
 * Changing assertion to check only existence
 * Install sagemaker-tensorflow from pypi. Add MKL environment variables for TF 1.8
 * get most recent saved model to export
 * pip install tensorflow 1.8 in 1.8 cpu image
 * install tensorflow extensions
 * upgrade cpu binaries in docker build
 * Force upgrade of the framework binaries to make sure the right binaries are installed.
 * Add Pillow to pip install list
 * Increase train steps for cifar distributed test to mitigate race condition
 * Add TensorFlow 1.8 dockerfiles
 * Add TensorFlow 1.7 dockerfiles
 * Explain how to download tf binaries from PyPI
 * Allow training without S3
 * Fix hyperparameter name for detecting a tuning job
 * Checkout v1.4.1 tag instead of r1.4 branch
 * Move processing of requirements file in.
 * Generate checkpoint path using TRAINING_JOB_NAME environment variable if needed
 * Wrap user-provided model_fn to pass arguments positionally (maintains compatibility with existing behavior)
 * Add more unit tests for trainer, fix __all__ and rename train.py to avoid import conflict
 * Use regional endpoint for S3 client
 * Update README.rst
 * Pass input_channels to eval_input_fn if defined
 * Fix setup.py to refer to renamed README
 * Add test and build instructions
 * Fix year in license headers
 * Add TensorFlow 1.6
 * Add test instructions in README
 * Add container support to install_requires
 * Add Apache license headers
 * Use wget to install tensorflow-model-server
 * Fix file path for integ test
 * Fix s3_prefix path in integ test
 * Fix typo in path for integ test
 * Add input_channels to train_input_fn interface.
 * Update logging and make serving_input_fn optional.
 * remove pip install in tensorflow training
 * Modify integration tests to run nvidia-docker for gpu
 * add h5py for keras models
 * Add local integ tests & resources
 * Restructure repo to use a directory per TF version for dockerfiles
 * Rename "feature_map" variables to "feature_dict" to avoid overloading it with the ML term "feature map"
 * Copying in changes from internal repo:
 * Add functional test
 * Fix FROM image names for final build dockerfiles
 * Add dockerfiles for building our production images (TF 1.4)
 * GPU Dockerfile and setup.py fixes
 * Add base image Dockerfiles for 1.4
 * Merge pull request #1 from aws/mvs-first-commit
 * first commit
 * Updating initial README.md from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Initial commit

## v0.1.0 (2019-05-22)

### Bug fixes and other changes

 * skip setup on second remote run
 * add setup file back
 * add branch name to remote gpu test run command
 * remove setup file in release build gpu test
 * ignore coverage in release build tests
 * use tar file name as framework_support_installable in build_all.py
 * Add release build
 * Explicitly set lower-bound for botocore version
 * Pull request to test codebuild trigger on TensorFlow script mode
 * Update integ test for checking Python version
 * Upgrade to TensorFlow 1.13.1
 * Add mpi4py to pip installs
 * Add SageMaker integ test for hyperparameter tuning model_dir logic
 * Add Horovod benchmark
 * Fix model_dir adjustment for hyperparameter tuning jobs
 * change model_dir to training job name if it is for tuning.
 * Tune test_s3_plugin test
 * Skip the s3_plugin test before new binary released
 * Add model saving warning at end of training
 * Specify region when creating S3 resource in integ tests
 * Fix instance_type fixture setup for tests
 * Read framework version from Python SDK for integ test default
 * Fix SageMaker Session handling in Horovod test
 * Configure encoding to be utf-8
 * Use the test argement framework_version in all tests
 * Fix broken test test_distributed_mnist_no_ps
 * Add S3 plugin tests
 * Skip horovod local CPU test in GPU instances
 * Add Horovod tests
 * Skip horovod integration tests
 * TensorFlow 1.12 and Horovod support
 * Deprecate get_marker. Use get_closest_marker instead
 * Force parameter server to run on CPU
 * Add python-dev and build-essential to Dockerfiles
 * Update script_mode_train_any_tf_script_in_sage_maker.ipynb
 * Skip keras local mode test on gpu and use random port for serving in the test
 * Fix Keras test
 * Create parameter server in different thread
 * Add Keras support
 * Fix broken unit tests
 * Unset CUDA_VISIBLE_DEVICES for worker processes
 * Disable GPU for parameter process
 * Set parameter process waiting to False
 * Update sagemaker containers
 * GPU fix
 * Set S3 environment variables
 * Add CI configuration files
 * Add distributed training support
 * Edited the tf script mode notebook
 * Add benchmarking script
 * Add Script Mode example
 * Add integration tests to run training jobs with sagemaker
 * Add tox.ini and configure coverage and flake runs
 * Scriptmode single machine training implementation
 * Update region in s3 boto client in serve
 * Update readme with instructions for 1.9.0 and above
 * Fix deserialization of dicts for json predict requests
 * Add dockerfile and update test for tensorflow 1.10.0
 * Support tensorflow 1.9.0
 * Add integ tests to verify that tensorflow in gpu-image can access gpu-devices.
 * train on 3 epochs for pipe mode test
 * Change error classes used by _default_input_fn() and _default_output_fn()
 * Changing assertion to check only existence
 * Install sagemaker-tensorflow from pypi. Add MKL environment variables for TF 1.8
 * get most recent saved model to export
 * pip install tensorflow 1.8 in 1.8 cpu image
 * install tensorflow extensions
 * upgrade cpu binaries in docker build
 * Force upgrade of the framework binaries to make sure the right binaries are installed.
 * Add Pillow to pip install list
 * Increase train steps for cifar distributed test to mitigate race condition
 * Add TensorFlow 1.8 dockerfiles
 * Add TensorFlow 1.7 dockerfiles
 * Explain how to download tf binaries from PyPI
 * Allow training without S3
 * Fix hyperparameter name for detecting a tuning job
 * Checkout v1.4.1 tag instead of r1.4 branch
 * Move processing of requirements file in.
 * Generate checkpoint path using TRAINING_JOB_NAME environment variable if needed
 * Wrap user-provided model_fn to pass arguments positionally (maintains compatibility with existing behavior)
 * Add more unit tests for trainer, fix __all__ and rename train.py to avoid import conflict
 * Use regional endpoint for S3 client
 * Update README.rst
 * Pass input_channels to eval_input_fn if defined
 * Fix setup.py to refer to renamed README
 * Add test and build instructions
 * Fix year in license headers
 * Add TensorFlow 1.6
 * Add test instructions in README
 * Add container support to install_requires
 * Add Apache license headers
 * Use wget to install tensorflow-model-server
 * Fix file path for integ test
 * Fix s3_prefix path in integ test
 * Fix typo in path for integ test
 * Add input_channels to train_input_fn interface.
 * Update logging and make serving_input_fn optional.
 * remove pip install in tensorflow training
 * Modify integration tests to run nvidia-docker for gpu
 * add h5py for keras models
 * Add local integ tests & resources
 * Restructure repo to use a directory per TF version for dockerfiles
 * Rename "feature_map" variables to "feature_dict" to avoid overloading it with the ML term "feature map"
 * Copying in changes from internal repo:
 * Add functional test
 * Fix FROM image names for final build dockerfiles
 * Add dockerfiles for building our production images (TF 1.4)
 * GPU Dockerfile and setup.py fixes
 * Add base image Dockerfiles for 1.4
 * Merge pull request #1 from aws/mvs-first-commit
 * first commit
 * Updating initial README.md from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Initial commit
