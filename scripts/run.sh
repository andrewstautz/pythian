#!/bin/bash
set -ex

source scripts/common.sh

docker run -it -p 8050:8050 -v $PWD:/src --rm stevemcquaid/$PACKAGE_NAME:latest python app.py

