FROM python:3.8-slim

RUN apt-get update

# Basic env setup
WORKDIR /opt

# Install requirements and module code
COPY requirements.txt /opt/requirements.txt
RUN apt-get install -y python3-gdcm
RUN python3 -m pip install -r /opt/requirements.txt

ENTRYPOINT ["python3", "run.py"]
CMD ["--help"]
