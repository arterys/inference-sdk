#!/bin/bash
CMD=$1
shift

docker build -t arterys_inference_server . && \
    docker run --rm -p 8900:8000 $@ arterys_inference_server $CMD
