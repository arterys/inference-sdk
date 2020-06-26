FROM ubuntu:16.04

RUN apt-get update && apt install -y software-properties-common  \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt-get install -y python3.7 python3-pip \
    && apt remove -y software-properties-common && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Basic env setup
WORKDIR /opt

# Install requirements and module code
COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install -r /opt/requirements.txt
COPY . /opt/

RUN which python3

ENTRYPOINT [ "python3", "mock_server.py" ]
