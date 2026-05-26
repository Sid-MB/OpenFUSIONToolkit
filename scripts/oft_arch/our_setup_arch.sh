#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${ROOT_PATH}"

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
OFT_MAKE_JOBS="${OFT_MAKE_JOBS:-8}"

case "${OFT_BUILD_FLAVOR}" in
  john)
    DEFAULT_ARCH_FLAGS="-O2 -mtune=generic -march=x86-64-v2"
    EXTRA_BUILD_LIBS_ARGS=(--oblas_dynamic_arch)
    ;;
  jag)
    DEFAULT_ARCH_FLAGS="-O2 -march=native"
    EXTRA_BUILD_LIBS_ARGS=()
    ;;
  *)
    echo "Unknown OFT_BUILD_FLAVOR=${OFT_BUILD_FLAVOR}; expected john or jag" >&2
    exit 2
    ;;
esac

OFT_ARCH_FLAGS="${OFT_ARCH_FLAGS:-${DEFAULT_ARCH_FLAGS}}"
export OFT_LIBS_FLAVOR="${OFT_BUILD_FLAVOR}"

echo "Building OFT flavor: ${OFT_BUILD_FLAVOR}"
echo "Dependency install suffix: _${OFT_LIBS_FLAVOR}"
echo "Arch flags: ${OFT_ARCH_FLAGS}"
echo "Make jobs: ${OFT_MAKE_JOBS}"

if [ "${CONDA_SHLVL:-0}" -gt 0 ]; then
  echo "You may want to consider deactivating conda."
fi

CC="${CC:-gcc}" \
CXX="${CXX:-g++}" \
FC="${FC:-gfortran}" \
python3 ./src/utilities/build_libs.py \
  --nthread="${OFT_MAKE_JOBS}" \
  --build_umfpack=1 \
  --build_arpack=1 \
  --opt_flags="${OFT_ARCH_FLAGS}" \
  "${EXTRA_BUILD_LIBS_ARGS[@]}"

bash config_cmake.sh
OFT_BUILD_FLAVOR="${OFT_BUILD_FLAVOR}" OFT_MAKE_JOBS="${OFT_MAKE_JOBS}" bash "${SCRIPT_DIR}/rebuild_arch.sh"
