FROM python:3.7-slim

# Install Python3-GDCM (dependency to read pixels of certain DICOM files)
RUN apt-get update && apt-get install -y python3-gdcm libglib2.0 libsm6 libxext6 libxrender-dev
RUN cp /usr/lib/python3/dist-packages/gdcm.py /usr/local/lib/python3.7/site-packages/ \
    && cp /usr/lib/python3/dist-packages/gdcmswig.py /usr/local/lib/python3.7/site-packages/ \
    && cp /usr/lib/python3/dist-packages/_gdcmswig*.so /usr/local/lib/python3.7/site-packages/ \
    && cp /usr/lib/x86_64-linux-gnu/libgdcm* /usr/local/lib/python3.7/site-packages/

# Basic env setup
WORKDIR /opt

# Create User that will run the script. 
# This is so that the output files are generated with the host user as owner
ARG USER_ID
ARG GROUP_ID

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} inference-user &&\
    useradd -l -u ${USER_ID} -g inference-user inference-user &&\
    install -d -m 0755 -o inference-user -g inference-user /home/inference-user \
;fi

# Install requirements and module code
COPY --chown=inference-user:inference-user requirements.txt /opt/requirements.txt

USER inference-user

RUN python3 -m pip install -r /opt/requirements.txt

ENTRYPOINT ["python3", "run.py"]
CMD ["--help"]
