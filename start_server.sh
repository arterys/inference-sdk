#!/bin/bash
docker build -t arterys_inference_server . && \
    docker run --rm -v $(pwd):/opt -p 8900:8000 -d arterys_inference_server $1
