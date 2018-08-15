FROM nvidia/cuda:9.0-base-ubuntu16.04

MAINTAINER Amazon AI

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl \
        git \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libcudnn7-dev=7.1.4.18-1+cuda9.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        vim \
        nginx \
        iputils-ping \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        h5py

# Set up grpc
RUN pip install enum34 futures mock six && \
    pip install --pre 'protobuf>=3.0.0a3' && \
    pip install -i https://testpypi.python.org/simple --pre grpcio

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release which works: https://github.com/bazelbuild/bazel/issues/4652
ENV BAZEL_VERSION 0.10.1
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7,6.1
ENV TF_CUDA_VERSION=9.0
ENV TF_CUDNN_VERSION=7
ENV CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

# TODO: upgrade to tf serving 1.8, which requires more work with updating
# dependencies. See current work in progress in tfserving-1.8 branch.
ENV TF_SERVING_VERSION=1.7.0

# Install tensorflow-serving-api
RUN pip install tensorflow-serving-api==$TF_SERVING_VERSION

# Download TensorFlow Serving
RUN cd / && git clone --recurse-submodules https://github.com/tensorflow/serving && \
  cd serving && \
  git checkout $TF_SERVING_VERSION

# Configure Tensorflow to use the GPU
WORKDIR /serving
RUN git clone --recursive https://github.com/tensorflow/tensorflow.git && \
  cd tensorflow && \
  git checkout v$TF_SERVING_VERSION && \
  tensorflow/tools/ci_build/builds/configured GPU

# Build TensorFlow Serving and Install it in /usr/local/bin
WORKDIR /serving
RUN bazel build -c opt --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    --crosstool_top=@local_config_cuda//crosstool:toolchain \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
    bazel clean --expunge

# Update libstdc++6, as required by tensorflow-serving >= 1.6: https://github.com/tensorflow/serving/issues/819
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y libstdc++6

# cleaning up the container
RUN rm -rf /serving && \
    rm -rf /bazel

WORKDIR /root

# Will install from pypi once packages are released there. For now, copy from local file system.
COPY $framework_installable .
COPY $framework_support_installable .

RUN framework_installable_local=$(basename $framework_installable) && \
    framework_support_installable_local=$(basename $framework_support_installable) && \
    \
    pip install --no-cache --upgrade $framework_installable_local && \
    pip install $framework_support_installable_local && \
    pip install "sagemaker-tensorflow>=1.9,<1.10" &&\
    \
    rm $framework_installable_local && \
    rm $framework_support_installable_local

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]