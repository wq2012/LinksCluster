#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add project modules to PYTHONPATH.
if [[ "${PYTHONPATH}" != *"${PROJECT_PATH}"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
fi

pushd ${PROJECT_PATH}

rm -f .coverage

PYTHON_BIN=$(command -v /opt/homebrew/bin/python3.11 || command -v python3)

# Run tests.
for TEST_FILE in $(find tests -name "*_test.py"); do
    echo "Running tests in ${TEST_FILE}"
    ${PYTHON_BIN} -m coverage run -a ${TEST_FILE}
done
echo "All tests passed!"

popd
