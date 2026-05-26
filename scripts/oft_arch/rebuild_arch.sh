#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

detect_build_flavor() {
  if [ -n "${OFT_BUILD_FLAVOR:-}" ]; then
    printf '%s\n' "${OFT_BUILD_FLAVOR}"
    return
  fi

  case "${SLURM_JOB_PARTITION:-}" in
    john*) printf 'john\n'; return ;;
    jag*) printf 'jag\n'; return ;;
  esac

  case "$(hostname -s)" in
    john*) printf 'john\n' ;;
    jag*|sc*) printf 'jag\n' ;;
    *) printf 'jag\n' ;;
  esac
}

OFT_BUILD_FLAVOR="$(detect_build_flavor)"
BUILD_DIR="${ROOT_PATH}/build_release_${OFT_BUILD_FLAVOR}"
OFT_MAKE_JOBS="${OFT_MAKE_JOBS:-8}"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "Missing build dir: ${BUILD_DIR}" >&2
  echo "Run first: OFT_BUILD_FLAVOR=${OFT_BUILD_FLAVOR} bash ${SCRIPT_DIR}/configure_cmake_arch.sh" >&2
  exit 2
fi

cd "${BUILD_DIR}"
make -j"${OFT_MAKE_JOBS}"
make install
