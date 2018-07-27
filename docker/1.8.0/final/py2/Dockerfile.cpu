# Use local version of image built from Dockerfile.cpu in /docker/1.8.0/base
FROM tensorflow-base:1.8.0-cpu-py2
MAINTAINER Amazon AI

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz

WORKDIR /root

COPY $framework_installable .
COPY $framework_support_installable .

RUN framework_installable_local=$(basename $framework_installable) && \
    framework_support_installable_local=$(basename $framework_support_installable) && \
    \
    pip install --no-cache --upgrade $framework_installable_local && \
    pip install $framework_support_installable_local && \
    pip install "sagemaker-tensorflow>=1.8,<1.9" &&\
    \
    rm $framework_installable_local && \
    rm $framework_support_installable_local

# Set environment variables for MKL
# TODO: investigate the right value for OMP_NUM_THREADS
ENV KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=1 KMP_SETTINGS=0

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]
