FROM python:3.10-slim

# Install Python3-GDCM (dependency to read pixels of certain DICOM files)
RUN apt-get update && apt-get install -y python3-gdcm libglib2.0 libsm6 libxext6 libxrender-dev gcc pkg-config libhdf5-dev

RUN cp /usr/lib/python3/dist-packages/gdcm.py /usr/local/lib/python3.10/site-packages/ \
    && cp /usr/lib/python3/dist-packages/gdcmswig.py /usr/local/lib/python3.10/site-packages/ \
    && cp /usr/lib/python3/dist-packages/_gdcmswig*.so /usr/local/lib/python3.10/site-packages/ \
    && cp /usr/lib/x86_64-linux-gnu/libgdcm* /usr/local/lib/python3.10/site-packages/

# Install requirements and module code
COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install -r /opt/requirements.txt

# Basic env setup
WORKDIR /opt
COPY . /opt/

ENTRYPOINT [ "python3", "mock_server.py" ]
