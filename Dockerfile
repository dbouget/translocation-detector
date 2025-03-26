# Creates virtual ubuntu in Docker image
FROM python:3.11-slim

# Creates GPU-compatible Docker environment - has to be adapted to the CUDA driver version on your GPU.
# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# maintainer of docker file
MAINTAINER David Bouget <david.bouget@sintef.no>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get -y install sudo
RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# downloading source code (not necessary, mostly to run the test scripts)
RUN git clone https://github.com/dbouget/translocation-detector.git

# Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -e .

RUN mkdir /workspace/resources

# CMD ["/bin/bash"]
ENTRYPOINT ["python3", "/workspace/translocation-detector/main.py"]




