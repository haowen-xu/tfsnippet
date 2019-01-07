#!/bin/bash

set -e

apt-get -y update && apt-get -y install unrar
pip install .
pip install -r requirements-dev.txt
mkdir -p /root/.config/matplotlib
echo 'backend : Agg' > /root/.config/matplotlib/matplotlibrc
export PYTHONPATH="$(pwd):${PYTHONPATH}"

if [ "${TENSORFLOW_VERSION}" = "*" ]; then
  python -m pytest \
      tests/utils/test_reuse.py \
      tests/utils/test_scope.py \
      tests/utils/test_session.py \
      tests/utils/test_shape_utils.py \
      tests/utils/test_tensor_wrapper.py \
      tests/utils/test_tfver.py \
      tests/utils/test_typeutils.py
else
  coverage run -m pytest && coveralls;
fi
