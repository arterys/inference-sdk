FROM tensorflow/tensorflow:2.5.1-gpu

WORKDIR /internal
COPY requirements.txt ./

# Workaround for:
# W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
# E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease' is no longer signed.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Use virtualenv to run python3.7 as
# other python versions give a "SystemError: unknown opcode" error
RUN apt-get update && apt-get install -y virtualenv python3.7
RUN virtualenv --python=/usr/bin/python3.7 venv
RUN . venv/bin/activate && pip install -r requirements.txt

WORKDIR /workdir
# Automatically enter the virtualenv when running the container
ENV PATH=/internal/venv/bin:$PATH

WORKDIR /opt
COPY . /opt/
ENTRYPOINT [ "python3", "mock_server.py" ]