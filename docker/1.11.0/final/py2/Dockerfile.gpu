FROM tensorflow/serving:1.11.0-gpu as tensorflow_serving_image
FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="Amazon AI"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ARG framework_installable
ARG framework_support_installable=sagemaker_tensorflow_container-1.0.0.tar.gz

ENV NCCL_VERSION=2.3.5-2+cuda9.0
ENV CUDNN_VERSION=7.3.1.20-1+cuda9.0
ENV TF_TENSORRT_VERSION=4.1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libgomp1 \
        python-dev \
        curl \
        nginx \
        && \
# The 'apt-get install' of nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0
# adds a new list which contains libnvinfer library, so it needs another
# 'apt-get update' to retrieve that list before it can actually install the
# library.
# We don't install libnvinfer-dev since we don't need to build against TensorRT,
# and libnvinfer4 doesn't contain libnvinfer.a static library.
    apt-get update && apt-get install -y --no-install-recommends \
        nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer4=${TF_TENSORRT_VERSION}-1+cuda9.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /usr/lib/x86_64-linux-gnu/libnvinfer_plugin* && \
    rm /usr/lib/x86_64-linux-gnu/libnvcaffe_parser* && \
    rm /usr/lib/x86_64-linux-gnu/libnvparsers*

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
COPY --from=tensorflow_serving_image /usr/bin/tensorflow_model_server /usr/bin/tensorflow_model_server


COPY $framework_installable .
COPY $framework_support_installable .

RUN pip install -U --no-cache-dir \
    numpy \
    scipy \
    sklearn \
    pandas \
    Pillow \
    h5py \
    tensorflow-serving-api==1.11.0 \
    \
    $framework_installable \
    $framework_support_installable \
    "sagemaker-tensorflow>=1.11,<1.12" && \
    \
    rm $framework_installable && \
    rm $framework_support_installable && \
    pip uninstall -y --no-cache-dir \ 
    markdown \
    tensorboard

# entry.py comes from sagemaker-container-support
ENTRYPOINT ["entry.py"]
