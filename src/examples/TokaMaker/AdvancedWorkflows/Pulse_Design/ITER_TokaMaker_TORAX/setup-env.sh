#!/usr/bin/env sh
# Source this file with `source setup-env.sh` from the ITER_TokaMaker_TORAX directory to create the local virtual environment, install dependencies from pyproject.toml, expose the OpenFUSIONToolkit install tree, and activate the environment in the current shell.

fail_message() {
    echo "$1" >&2
}

PROJECT_DIR="$(pwd -P)"
OFT_ROOT="$(CDPATH= cd -- "${PROJECT_DIR}/../../../../../../" && pwd -P)"
VENV_DIR="${PROJECT_DIR}/.venv"
REBUILD_SCRIPT="${OFT_ROOT}/rebuild.sh"
OFT_PYTHONPATH="${OFT_ROOT}/install_release/python"
REQUIREMENTS_FILE="${VENV_DIR}/pyproject-requirements.txt"
PIP_CACHE_DIR="${VENV_DIR}/pip-cache"

if [ ! -x "${REBUILD_SCRIPT}" ]; then
    fail_message "OpenFUSIONToolkit rebuild script not found or not executable at ${REBUILD_SCRIPT}"
    return 1 2>/dev/null || exit 1
fi

if ! "${REBUILD_SCRIPT}" >/dev/null; then
    fail_message "OpenFUSIONToolkit rebuild failed"
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "${OFT_PYTHONPATH}/OpenFUSIONToolkit" ]; then
    fail_message "OpenFUSIONToolkit package not found at ${OFT_PYTHONPATH}"
    return 1 2>/dev/null || exit 1
fi

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    fail_message "python3.12 or python3 is required to create .venv"
    return 1 2>/dev/null || exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
        fail_message "Failed to create ${VENV_DIR}"
        return 1 2>/dev/null || exit 1
    fi
fi

"${VENV_DIR}/bin/python" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ is required by pyproject.toml; recreate .venv with python3.12 available on PATH")
PY
if [ "$?" -ne 0 ]; then
    fail_message "The virtual environment does not satisfy the Python version required by pyproject.toml"
    return 1 2>/dev/null || exit 1
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    if ! "${VENV_DIR}/bin/python" -m ensurepip --upgrade; then
        fail_message "Failed to bootstrap pip in ${VENV_DIR}"
        return 1 2>/dev/null || exit 1
    fi
fi

mkdir -p "${PIP_CACHE_DIR}"

if ! "${VENV_DIR}/bin/python" - "${REQUIREMENTS_FILE}" <<'PY'
import tomllib
import sys
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
Path(sys.argv[1]).write_text("\n".join(project.get("dependencies", [])) + "\n")
PY
then
    fail_message "Failed to read dependencies from pyproject.toml"
    return 1 2>/dev/null || exit 1
fi

if ! PIP_CACHE_DIR="${PIP_CACHE_DIR}" "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"; then
    fail_message "Failed to install dependencies from pyproject.toml"
    return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${OFT_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
. "${VENV_DIR}/bin/activate"

echo "Environment ready. Python: $(python --version 2>&1)"
echo "PYTHONPATH includes: ${OFT_PYTHONPATH}"
