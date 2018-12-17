#!/bin/bash


if [[ "${TRAVIS_BRANCH}" = "master" || "${TRAVIS_BRANCH}" = "develop" ]]; then
  if [[ "${CORE_TESTS_ONLY}" = "1" ]]; then
    export RUN_EXAMPLES_TEST_CASE=0;
  else
    export RUN_EXAMPLES_TEST_CASE=1;
  fi
fi

docker run
    -v "$(pwd)":"$(pwd)"
    -v "/home/travis/.tfsnippet":"/root/.tfsnippet"
    -v "/home/travis/.keras":"/root/.keras"
    -w "$(pwd)"
    -e TRAVIS="${TRAVIS}"
    -e TRAVIS_JOB_ID="${TRAVIS_JOB_ID}"
    -e TRAVIS_BRANCH="${TRAVIS_BRANCH}"
    -e CORE_TESTS_ONLY="${CORE_TESTS_ONLY}"
    -e RUN_EXAMPLES_TEST_CASE="${RUN_EXAMPLES_TEST_CASE}"
    "haowenxu/travis-tensorflow-docker:py${PYTHON_VERSION}tf${TENSORFLOW_VERSION}"
    bash "scripts/travis-docker-entry.sh"
