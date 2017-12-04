#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to install dependencies of this project."""

import codecs
import os
import re
import subprocess
import sys

script_root = os.path.abspath(os.path.split(__file__)[0])

# detect the environment
try:
    ANACONDA_SIGNATURE = (
        'Anaconda',
        'Continuum',
    )
    subprocess.check_call(['conda', '--version'])
    is_anaconda = any(s in sys.version for s in ANACONDA_SIGNATURE)
except Exception:
    is_anaconda = False

if 'TF_DEVICE' in os.environ:
    device = os.environ['TF_DEVICE']
else:
    try:
        subprocess.check_call(['nvcc', '--version'])
        device = 'gpu'
    except Exception:
        device = 'cpu'

# Read dependencies from requirements.txt
tf_deps = []  # which will be installed after TensorFlow is installed
conda_deps = []
pip_deps = []
active_deps = pip_deps
requirements_file = os.path.join(script_root, '../requirements-dev.txt')

if sys.version_info[0] == 2:
    f = open(requirements_file, 'rb')
else:
    f = codecs.open(requirements_file, 'rb', 'utf-8')

with f:
    for line in f:
        line = line.strip()
        if line.startswith('# tensorFlow'):
            active_deps = tf_deps
        elif line.startswith('# anaconda'):
            active_deps = conda_deps
        elif line.startswith('# pip'):
            active_deps = pip_deps
        elif line and not line.startswith('#'):
            # there might be whitespaces in the line, and we want to
            # throw away all these characters.
            active_deps.append(''.join(line.split()))

# If the python is Anaconda based, install some packages via conda
if conda_deps:
    if is_anaconda:
        subprocess.check_call(['conda', 'install', '--yes', '-q'] + conda_deps)
    else:
        pip_deps = conda_deps + pip_deps

# Install dependencies via pip
if pip_deps:
    subprocess.check_call(['python', '-m', 'pip', 'install'] + pip_deps)

# Install TensorFlow
tf_package = 'tensorflow-gpu' if device == 'gpu' else 'tensorflow'
subprocess.check_call(['python', '-m', 'pip', 'install', tf_package])

# Install dependencies after TensorFlow
if tf_deps:
    tf_deps = list(filter(
        lambda s: not re.match(r'^tensorflow(-gpu)?$', s, re.I),
        tf_deps
    ))
    subprocess.check_call(['python', '-m', 'pip', 'install'] + tf_deps)
