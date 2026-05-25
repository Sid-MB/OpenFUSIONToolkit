#!/usr/bin/env bash
# The first time, use ./our_setup.sh, after you can use this to quickly rebuild.
set -euo pipefail

cd "$(dirname "$0")/build_release"

make -j8
make install
