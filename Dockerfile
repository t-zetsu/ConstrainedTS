ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && apt-get install -y git wget \
  && apt update 

WORKDIR /work
COPY requirements.txt /work

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV PYTHONIOENCODING utf-8
ENV CUDA_HOME /usr/local/cuda-11.3

