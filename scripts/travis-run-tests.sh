#!/bin/bash

function runTest() {
  PY_VER="$1"
  TF_VER="$2"
  RUN_EXAMPLES_TEST_CASE="$3"

  echo "TFSnippet Tests

    CORE_TESTS_ONLY=${CORE_TESTS_ONLY}
    PYTHON_VERSION=${PY_VER}
    TENSORFLOW_VERSION=${TF_VER}
    RUN_EXAMPLES_TEST_CASE=${RUN_EXAMPLES_TEST_CASE}
  "

  IMAGE_NAME="haowenxu/travis-tensorflow-docker:py${PY_VER}tf${TF_VER}"
  docker pull "${IMAGE_NAME}"
  docker run \
      -v "$(pwd)":"$(pwd)" \
      -v "/home/travis/.tfsnippet":"/root/.tfsnippet" \
      -v "/home/travis/.keras":"/root/.keras" \
      -w "$(pwd)" \
      -e TRAVIS="${TRAVIS}" \
      -e TRAVIS_JOB_ID="${TRAVIS_JOB_ID}" \
      -e TRAVIS_BRANCH="${TRAVIS_BRANCH}" \
      -e CORE_TESTS_ONLY="${CORE_TESTS_ONLY}" \
      -e RUN_EXAMPLES_TEST_CASE="${RUN_EXAMPLES_TEST_CASE}" \
      "${IMAGE_NAME}" \
      bash "scripts/travis-docker-entry.sh"
}
if [[ "${CORE_TESTS_ONLY}" = "1" ]]; then
  for PY_VER in 2 3; do
    for TF_VER in 1.5 1.6 1.7 1.8 1.9 1.10 1.11 1.12; do
      runTest "${PY_VER}" "${TF_VER}" "0";
    done
  done
else
  if [[ "${TRAVIS_BRANCH}" = "master" || "${TRAVIS_BRANCH}" = "develop" ]]; then
    runTest "${PYTHON_VERSION}" "${TENSORFLOW_VERSION}" "1";
  else
    runTest "${PYTHON_VERSION}" "${TENSORFLOW_VERSION}" "0";
  fi
fi
