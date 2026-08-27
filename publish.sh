#!/bin/bash
set -o errexit

# This script requires these tools:
# pip3 install --user --upgrade setuptools wheel
# pip3 install --user --upgrade twine

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# clean up
rm -rf build
rm -rf dist
rm -rf linkscluster.egg-info

PYTHON_BIN=$(command -v /opt/homebrew/bin/python3.11 || command -v python3)

# build and upload
${PYTHON_BIN} setup.py sdist bdist_wheel
${PYTHON_BIN} -m twine upload dist/* --verbose

popd
