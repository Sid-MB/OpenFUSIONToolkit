#!/bin/bash
# Build instructions https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/wiki/Building-OFT-on-macOS
set -euxo pipefail

cd "$(dirname "$0")"
if [ "${CONDA_SHLVL:-0}" -gt 0 ]; then # Deactivate conda if enabled so Anaconda tools do not shadow the macOS build toolchain?
    [ -f "$(conda info --base)/etc/profile.d/conda.sh" ] && source "$(conda info --base)/etc/profile.d/conda.sh"
    # conda deactivate
    echo "You may want to consider deactivating conda."
fi
# If you are getting a warning about fortran not being enabled you may need to run `conda activate openfusion`. Or, if that doesn't exist: `conda create -n openfusion -c conda-forge python=3.12 gfortran` and then `conda activate openfusion`.

# Siddharth needs the following line on his machine, probably not necessary for anyone else
export PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
CC=gcc-15 CXX=g++-15 FC=gfortran-15 python3 ./src/utilities/build_libs.py --nthread=8 --build_umfpack=1 --build_arpack=1
bash config_cmake.sh

bash rebuild.sh
