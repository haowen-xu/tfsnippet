#!/bin/bash

if [[ "${TRAVIS_BRANCH}" = "master" || "${TRAVIS_BRANCH}" = "develop" ]]; then
    export RUN_EXAMPLES_TEST_CASE=1
fi

apt-get -y update &&  \
    apt-get -y install unrar &&  \
    pip install . &&  \
    pip install -r requirements-dev.txt && \
    mkdir -p /root/.config/matplotlib &&  \
    echo 'backend : Agg' > /root/.config/matplotlib/matplotlibrc && \
    coverage run -m py.test &&  \
    coveralls
