FROM ubuntu:16.04

MAINTAINER Amazon AI

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz
ARG tensorflow_model_server

WORKDIR /root

COPY $framework_installable .
COPY $framework_support_installable .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
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
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        wget \
        vim \
        iputils-ping \
        nginx \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

# TODO: upgrade to tf serving 1.8, which requires more work with updating
# dependencies. See current work in progress in tfserving-1.8 branch.
ENV TF_SERVING_VERSION=1.7.0

RUN pip install numpy boto3 six awscli flask==0.11 Jinja2==2.9 tensorflow-serving-api==$TF_SERVING_VERSION gevent gunicorn

# Install TF Serving pkg
COPY $tensorflow_model_server /usr/bin/tensorflow_model_server

# Update libstdc++6, as required by tensorflow-serving >= 1.6: https://github.com/tensorflow/serving/issues/819
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y libstdc++6

RUN framework_installable_local=$(basename $framework_installable) && \
    framework_support_installable_local=$(basename $framework_support_installable) && \
    \
    pip install --no-cache --upgrade $framework_installable_local && \
    pip install $framework_support_installable_local && \
    pip install "sagemaker-tensorflow>=1.10,<1.11" &&\
    \
    rm $framework_installable_local && \
    rm $framework_support_installable_local

# Set environment variables for MKL
# TODO: investigate the right value for OMP_NUM_THREADS
ENV KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=1 KMP_SETTINGS=0

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]
