FROM ubuntu:16.04

LABEL maintainer="Amazon AI"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz
ARG tensorflow_model_server

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python-dev \
    curl \
    nginx \
 && rm -rf /var/lib/apt/lists/*

# Python won’t try to write .pyc or .pyo files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Removing tests to free some space
RUN set -ex; \
	curl -k -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		"pip==18.1" \
	; \
	pip --version; \
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
	rm -f get-pip.py

WORKDIR /

# Install TF Serving pkg
COPY $tensorflow_model_server /usr/bin/tensorflow_model_server

COPY $framework_installable .
COPY $framework_support_installable .

RUN pip install -U --no-cache-dir \
    numpy \
    scipy \
    sklearn \
    pandas \
    Pillow \
    h5py \
    tensorflow-serving-api==1.12.0 \
    \
    $framework_installable \
    $framework_support_installable \
    "sagemaker-tensorflow>=1.12,<1.13" && \
    \
    rm $framework_installable && \
    rm $framework_support_installable && \
    pip uninstall -y --no-cache-dir \
    markdown \
    tensorboard

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]
