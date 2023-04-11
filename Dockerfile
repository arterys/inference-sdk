FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /internal
COPY requirements.txt ./

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
ENTRYPOINT [ "python3", "strain_inference_server.py" ]