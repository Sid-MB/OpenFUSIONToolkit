#!/bin/bash
# Build instructions https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/wiki/Building-OFT-on-macOS
# Siddharth uses this but no need to use it if what you're doing is working.
set -euxo pipefail

cd "$(dirname "$0")"
if [ "${CONDA_SHLVL:-0}" -gt 0 ]; then # Deactivate conda if enabled so Anaconda tools do not shadow the macOS build toolchain?
    [ -f "$(conda info --base)/etc/profile.d/conda.sh" ] && source "$(conda info --base)/etc/profile.d/conda.sh"
    # conda deactivate
    echo "You may want to consider deactivating conda."
fi
# If you are getting a warning about fortran not being enabled you may need to run `conda activate openfusion`. Or, if that doesn't exist: `conda create -n openfusion -c conda-forge python=3.12 gfortran` and then `conda activate openfusion`.

# Pick a compiler toolchain depending on platform.
# On macOS (Homebrew) the compilers are versioned (gcc-15/g++-15/gfortran-15);
# on Linux we use the unversioned names provided by the conda env / system.
if [ "$(uname)" = "Darwin" ]; then
    # Siddharth needs the following line on his machine, probably not necessary for anyone else
    export PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
    CC=gcc-15
    CXX=g++-15
    FC=gfortran-15
else
    CC=gcc
    CXX=g++
    FC=gfortran
    # We build with the conda toolchain (gcc/gfortran from $CONDA_PREFIX). CMake's
    # FindOpenMP otherwise picks up the *system* libgomp (e.g. /usr/lib/gcc/.../13),
    # which is built against a newer glibc and fails to link ("undefined reference
    # to dlerror@GLIBC_2.34"). Force CMake to find the conda libgomp first and embed
    # an rpath so the binaries resolve it at runtime.
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export CMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
        export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib ${LDFLAGS:-}"
    fi
fi
CC="$CC" CXX="$CXX" FC="$FC" python3 ./src/utilities/build_libs.py --nthread=8 --build_umfpack=1 --build_arpack=1
bash config_cmake.sh

bash rebuild.sh
