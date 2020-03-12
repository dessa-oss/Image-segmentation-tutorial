FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN  apt-get update \ 
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update\
    && apt-get upgrade -y \
    && apt-get install bzip2 

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        && rm /tmp/requirements.txt

ENTRYPOINT ["python"]
