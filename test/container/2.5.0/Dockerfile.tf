FROM tensorflow/tensorflow:2.5.0-gpu

ENV SAGEMAKER_TRAINING_MODULE sagemaker_tensorflow_container.training:main

COPY dist/sagemaker_tensorflow_training-*.tar.gz /sagemaker_tensorflow_training.tar.gz
RUN pip install --upgrade --no-cache-dir /sagemaker_tensorflow_training.tar.gz && \
    rm /sagemaker_tensorflow_training.tar.gz
