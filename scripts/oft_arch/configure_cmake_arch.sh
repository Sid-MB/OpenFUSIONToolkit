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
BUILD_DIR="${ROOT_PATH}/build_release_${OFT_BUILD_FLAVOR}"
INSTALL_DIR="${ROOT_PATH}/install_release_${OFT_BUILD_FLAVOR}"

case "${OFT_BUILD_FLAVOR}" in
  john)
    DEFAULT_ARCH_FLAGS="-O2 -mtune=generic -march=x86-64-v2"
    ;;
  jag)
    DEFAULT_ARCH_FLAGS="-O2 -march=native"
    ;;
  *)
    echo "Unknown OFT_BUILD_FLAVOR=${OFT_BUILD_FLAVOR}; expected john or jag" >&2
    exit 2
    ;;
esac

OFT_ARCH_FLAGS="${OFT_ARCH_FLAGS:-${DEFAULT_ARCH_FLAGS}}"
OFT_MAKE_JOBS="${OFT_MAKE_JOBS:-8}"

dep_root() {
  local name="$1"
  if [ -d "${ROOT_PATH}/${name}_${OFT_BUILD_FLAVOR}" ]; then
    printf '%s\n' "${ROOT_PATH}/${name}_${OFT_BUILD_FLAVOR}"
  else
    printf '%s\n' "${ROOT_PATH}/${name}"
  fi
}

METIS_ROOT="$(dep_root metis-5_1_0)"
HDF5_ROOT="$(dep_root hdf5-1_14_6)"
OPENBLAS_ROOT="$(dep_root OpenBLAS-0_3_30)"
ARPACK_ROOT="$(dep_root arpack-ng-3_9_1)"
LIBXML2_ROOT="$(dep_root libxml2-v2_15_2)"
UMFPACK_ROOT="$(dep_root UMFPACK-6_3_5)"

echo "Configuring OFT flavor: ${OFT_BUILD_FLAVOR}"
echo "Build dir: ${BUILD_DIR}"
echo "Install dir: ${INSTALL_DIR}"
echo "Arch flags: ${OFT_ARCH_FLAGS}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX:PATH="${INSTALL_DIR}" \
  -DOFT_BUILD_TESTS:BOOL=FALSE \
  -DOFT_PY_KERNEL:STRING=python3 \
  -DOFT_BUILD_EXAMPLES:BOOL=FALSE \
  -DOFT_BUILD_PYTHON:BOOL=TRUE \
  -DOFT_BUILD_DOCS:BOOL=FALSE \
  -DOFT_USE_OpenMP:BOOL=TRUE \
  -DOFT_PACKAGE_BUILD:BOOL=FALSE \
  -DOFT_PACKAGE_NIGHTLY:BOOL=TRUE \
  -DOFT_COVERAGE:BOOL=FALSE \
  -DOFT_DEBUG_STACK:BOOL=FALSE \
  -DOFT_PROFILING:BOOL=FALSE \
  -DOFT_THINCURR_LEGACY:BOOL=FALSE \
  -DCMAKE_C_COMPILER:FILEPATH="${CC:-gcc}" \
  -DCMAKE_CXX_COMPILER:FILEPATH="${CXX:-g++}" \
  -DCMAKE_Fortran_COMPILER:FILEPATH="${FC:-gfortran}" \
  -DCMAKE_C_FLAGS:STRING="${OFT_ARCH_FLAGS}" \
  -DCMAKE_CXX_FLAGS:STRING="${OFT_ARCH_FLAGS}" \
  -DCMAKE_Fortran_FLAGS:STRING="-fallow-argument-mismatch ${OFT_ARCH_FLAGS}" \
  -DOFT_USE_MPI:BOOL=FALSE \
  -DOFT_METIS_ROOT:PATH="${METIS_ROOT}" \
  -DHDF5_ROOT:PATH="${HDF5_ROOT}" \
  -DBLAS_ROOT:PATH="${OPENBLAS_ROOT}" \
  -DLAPACK_ROOT:PATH="${OPENBLAS_ROOT}" \
  -DBLA_VENDOR:STRING=OpenBLAS \
  -DOFT_ARPACK_ROOT:PATH="${ARPACK_ROOT}" \
  -DLIBXML2_ROOT:PATH="${LIBXML2_ROOT}" \
  -DOFT_UMFPACK_ROOT:PATH="${UMFPACK_ROOT}" \
  "${ROOT_PATH}/src"

echo "Configured ${OFT_BUILD_FLAVOR}. Build with:"
echo "  OFT_BUILD_FLAVOR=${OFT_BUILD_FLAVOR} bash ${SCRIPT_DIR}/rebuild_arch.sh"
