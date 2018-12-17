#!/bin/bash

set -e

apt-get -y update && apt-get -y install unrar
pip install .
pip install -r requirements-dev.txt
mkdir -p /root/.config/matplotlib
echo 'backend : Agg' > /root/.config/matplotlib/matplotlibrc

if [ "${TENSORFLOW_VERSION}" = "*" ]; then
  coverage run -m py.test \
      tests/utils/test_reuse.py \
      tests/utils/test_scope.py \
      tests/utils/test_session.py \
      tests/utils/test_shape_utils.py \
      tests/utils/test_tensor_wrapper.py \
      tests/utils/test_tfver.py \
      tests/utils/test_typeutils.py \
    && coveralls;
else
  coverage run -m py.test && coveralls;
fi
