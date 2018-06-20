FROM ubuntu:16.04

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

WORKDIR /root

RUN pip install numpy boto3 six awscli flask==0.11 Jinja2==2.9 tensorflow-serving-api==1.5 gevent gunicorn

# install tensorflow-model-server 1.5. 1.6 is not working as of 3/29/2018 for unknown reasons.
RUN wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server/t/tensorflow-model-server/tensorflow-model-server_1.5.0_all.deb' && \
    dpkg -i tensorflow-model-server_1.5.0_all.deb
