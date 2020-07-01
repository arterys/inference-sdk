FROM arterys/inference-sdk-base:0.1-cpu

# Install requirements and module code
COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install -r /opt/requirements.txt

# Basic env setup
WORKDIR /opt
COPY . /opt/

ENTRYPOINT [ "python3", "mock_server.py" ]
