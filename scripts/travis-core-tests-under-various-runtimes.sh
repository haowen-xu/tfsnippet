#!/bin/bash

for PYTHON_VERSION in 2.7 3.5; do
  for TENSORFLOW_VERSION in 1.5 1.6 1.7 1.8 1.9 1.10 1.11 1.12; do
  (
    export PYTHON_VERSION
    export TENSORFLOW_VERSION
    bash scripts/travis-run-tests.sh
  );
  done
done
