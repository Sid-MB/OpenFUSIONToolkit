# Architecture-Specific OFT Builds

The run wrappers source:

```bash
../../../../../../scripts/oft_arch/select_oft_install.sh
```

The selector chooses an OFT install from the Slurm partition or hostname:

- `john` partition/hosts: `install_release_john`
- `jag-standard` or `jag*` hosts: `install_release_jag` if present, otherwise `install_release`

Override selection when needed:

```bash
OFT_INSTALL_FLAVOR=john
OFT_INSTALL_DIR=/path/to/install_release_custom
```

## Fresh Build

Build the matching install before submitting jobs:

```bash
OFT_BUILD_FLAVOR=john OFT_MAKE_JOBS=8 bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
OFT_BUILD_FLAVOR=jag OFT_MAKE_JOBS=8 bash ../../../../../../scripts/oft_arch/our_setup_arch.sh
```

## Reconfigure/Rebuild Only

If third-party libraries already exist and only OFT needs to be regenerated:

```bash
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/configure_cmake_arch.sh
OFT_BUILD_FLAVOR=john bash ../../../../../../scripts/oft_arch/rebuild_arch.sh
```

Use `OFT_BUILD_FLAVOR=jag` for the corresponding GPU-node install.
