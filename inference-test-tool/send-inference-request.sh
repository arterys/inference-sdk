#!/bin/bash

if [[ "$(docker images -q test_arterys_sdk 2> /dev/null)" == "" ]]; then
    docker image build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) -t test_arterys_sdk . 
fi

docker run --network host --rm -v $(pwd):/opt -it test_arterys_sdk $@
