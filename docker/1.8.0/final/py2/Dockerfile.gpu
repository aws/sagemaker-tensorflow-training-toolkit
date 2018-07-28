# Use local version of image built from Dockerfile.gpu in /docker/1.8.0/base
FROM tensorflow-base:1.8.0-gpu-py2
MAINTAINER Amazon AI

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz

WORKDIR /root

# Will install from pypi once packages are released there. For now, copy from local file system.
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

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]
