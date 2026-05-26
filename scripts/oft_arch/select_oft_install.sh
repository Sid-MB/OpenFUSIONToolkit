#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OFT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
OFT_SCRIPT_IN_INSTALL=0
if [ -d "${OFT_ROOT}/python/OpenFUSIONToolkit" ]; then
  OFT_SCRIPT_IN_INSTALL=1
fi

detect_oft_flavor() {
  if [ -n "${OFT_INSTALL_FLAVOR:-}" ]; then
    printf '%s\n' "${OFT_INSTALL_FLAVOR}"
    return
  fi

  case "${SLURM_JOB_PARTITION:-}" in
    john*)
      printf 'john\n'
      return
      ;;
    jag*)
      printf 'jag\n'
      return
      ;;
  esac

  case "$(hostname -s)" in
    john*)
      printf 'john\n'
      ;;
    jag*|sc*)
      printf 'jag\n'
      ;;
    *)
      printf 'jag\n'
      ;;
  esac
}

OFT_SELECTED_FLAVOR="$(detect_oft_flavor)"

if [ -n "${OFT_INSTALL_DIR:-}" ]; then
  OFT_SELECTED_INSTALL="${OFT_INSTALL_DIR}"
elif [ "${OFT_SCRIPT_IN_INSTALL}" -eq 1 ]; then
  OFT_SELECTED_INSTALL="${OFT_ROOT}"
else
  case "${OFT_SELECTED_FLAVOR}" in
    john)
      OFT_SELECTED_INSTALL="${OFT_ROOT}/install_release_john"
      ;;
    jag)
      if [ -d "${OFT_ROOT}/install_release_jag/python/OpenFUSIONToolkit" ]; then
        OFT_SELECTED_INSTALL="${OFT_ROOT}/install_release_jag"
      else
        OFT_SELECTED_INSTALL="${OFT_ROOT}/install_release"
      fi
      ;;
    *)
      echo "Unknown OFT install flavor: ${OFT_SELECTED_FLAVOR}" >&2
      return 2 2>/dev/null || exit 2
      ;;
  esac
fi

if [ ! -d "${OFT_SELECTED_INSTALL}/python/OpenFUSIONToolkit" ]; then
  cat >&2 <<EOF
Missing OFT install for flavor '${OFT_SELECTED_FLAVOR}':
  ${OFT_SELECTED_INSTALL}

Build it first, for example:
  OFT_BUILD_FLAVOR=${OFT_SELECTED_FLAVOR} bash ${SCRIPT_DIR}/our_setup_arch.sh

Or, if the flavor-specific third-party libraries already exist:
  OFT_BUILD_FLAVOR=${OFT_SELECTED_FLAVOR} bash ${SCRIPT_DIR}/configure_cmake_arch.sh
  OFT_BUILD_FLAVOR=${OFT_SELECTED_FLAVOR} bash ${SCRIPT_DIR}/rebuild_arch.sh
EOF
  return 2 2>/dev/null || exit 2
fi

export OFT_SELECTED_FLAVOR
export OFT_SELECTED_INSTALL
export PYTHONPATH="${OFT_SELECTED_INSTALL}/python${PYTHONPATH:+:${PYTHONPATH}}"

if [ -d "${OFT_SELECTED_INSTALL}/lib" ]; then
  export LD_LIBRARY_PATH="${OFT_SELECTED_INSTALL}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [ -d "${OFT_SELECTED_INSTALL}/bin" ]; then
  export LD_LIBRARY_PATH="${OFT_SELECTED_INSTALL}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PATH="${OFT_SELECTED_INSTALL}/bin${PATH:+:${PATH}}"
fi
