# Use local version of image built from Dockerfile.cpu in /docker/base
FROM tensorflow-base:1.4.1-cpu-py2
MAINTAINER Amazon AI

WORKDIR /root

RUN apt-get -y update && \
    apt-get -y install curl && \
    apt-get -y install wget && \
    apt-get -y install vim && \
    apt-get -y install iputils-ping && \
    apt-get -y install nginx

RUN pip install numpy boto3 six awscli flask==0.11 Jinja2==2.9 tensorflow-serving-api==1.4 gevent gunicorn

# Install tensorflow-model-server 1.4.
RUN wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server/t/tensorflow-model-server/tensorflow-model-server_1.4.0_all.deb' && \
    dpkg -i tensorflow-model-server_1.4.0_all.deb

# install telegraf
RUN cd /tmp && \
    curl -O https://dl.influxdata.com/telegraf/releases/telegraf_1.4.2-1_amd64.deb && \
    dpkg -i telegraf_1.4.2-1_amd64.deb && \
    cd -

COPY sagemaker_tensorflow_container-1.0.0.tar.gz .

RUN pip install sagemaker_tensorflow_container-1.0.0.tar.gz

RUN rm sagemaker_tensorflow_container-1.0.0.tar.gz

ENTRYPOINT ["entry.py"]
