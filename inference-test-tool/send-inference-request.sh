#!/bin/bash

docker image build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) -t test_arterys_sdk .
docker run --network host --rm -v $(pwd):/opt test_arterys_sdk $@
