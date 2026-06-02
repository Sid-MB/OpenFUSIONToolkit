
- Use `uv run python`, not `python`.
- `ModuleNotFoundError: No module named 'OpenFUSIONToolkit'` means that OpenFusionToolkit is not on `PYTHONPATH`. Add it. It's located [here](../../../../../python/OpenFUSIONToolkit/).
- You can change files in this git repo outside of the ITER_TokaMaker_TORAX directory, most notably [`pulse_design.py`](../../../../../python/OpenFUSIONToolkit/TokaMaker/pulse_design.py). When you change it, if changes aren't showing up, also run [`rebuild.sh`](../../../../../../../rebuild.sh). DO NOT EDIT files in `install_release/` or `build_release/` directly.

(Relative paths are from `ITER_TokaMaker_TORAX/`.)
