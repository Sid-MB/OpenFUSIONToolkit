"""Offline postprocessing for actor evaluations."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from log import get_logger

logger = get_logger(__name__)


def postprocess_actor_eval(
    result,
    output_dir,
    render_plots=True,
    render_movie=True,
    render_summary=True,
    tmtx=None,
    movie_speed_factor=10.0,
):
    output_dir = Path(output_dir).resolve()
    outputs_dir = output_dir / "artifacts"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "actor_eval_summary.json"
    if render_summary and result_path.exists():
        logger.info("Summary available at %s", result_path)

    if not render_plots and not render_movie:
        return {"result_path": str(result_path)}

    if tmtx is None:
        bundle_path = result.get("bundle_path") if isinstance(result, dict) else None
        if bundle_path and Path(bundle_path).exists():
            try:
                with Path(bundle_path).open("rb") as f:
                    bundle = pickle.load(f)
                tmtx = _bundle_to_tmtx_shim(bundle)
            except Exception as exc:
                logger.warning("Bundle reconstruction failed: %s", exc)
                tmtx = None
    if tmtx is None:
        try:
            from collect_trajectories_delta import configure_tmtx, setup_tokamaker
            from collect_trajectories_delta import resolve_seed_eqdsk_paths, default_relax_geometry
            from collect_trajectories_delta import patch_initial_relax_cache_loader
        except Exception as exc:
            logger.warning("Postprocess skipped: %s", exc)
            return {"result_path": str(result_path), "warning": str(exc)}

        cwd = Path.cwd()
        eqdsk_list = resolve_seed_eqdsk_paths(str(cwd))
        geom = default_relax_geometry()
        mygs, _, _, _ = setup_tokamaker(str(cwd))
        tmtx = configure_tmtx(
            mygs,
            [],
            eqdsk_list,
            geom["eqtimes"],
            geom["coil_bounds"],
            geom["x_points"],
            geom["Ip_targets"],
            geom["ne_init"],
            geom["Te_init"],
            geom["psi_sample"],
            grid_size=int(result.get("metrics", {}).get("actor_eval/grid_size", 51) or 51),
        )
        patch_initial_relax_cache_loader(tmtx)

    generated = {}
    if render_summary:
        try:
            generated["summary"] = tmtx.summary()
        except Exception as exc:
            logger.warning("summary() failed during postprocess: %s", exc)
    if render_plots:
        for name in ("plot_scalars", "plot_lcfs_evolution", "plot_profile_evolution"):
            fn = getattr(tmtx, name, None)
            if fn is None:
                continue
            try:
                generated[name] = fn(display=False, save_path=str(outputs_dir / f"{name}.png"))
            except TypeError:
                generated[name] = fn()
            except Exception as exc:
                logger.warning("%s failed during postprocess: %s", name, exc)
    if render_movie:
        fn = getattr(tmtx, "make_movie", None)
        if fn is not None:
            try:
                generated["movie"] = fn(
                    notebook_mode=False,
                    speed_factor=movie_speed_factor,
                    save_path=str(outputs_dir / "movie.mp4"),
                )
            except TypeError:
                generated["movie"] = fn(notebook_mode=False, speed_factor=movie_speed_factor)
            except Exception as exc:
                logger.warning("make_movie failed during postprocess: %s", exc)

    if result_path.exists():
        try:
            import subprocess, sys
            plot_script = Path(__file__).resolve().parent.parent / "visualize" / "plot_heating_schedule.py"
            subprocess.run(
                [sys.executable, str(plot_script), str(outputs_dir), str(result_path)],
                check=True,
            )
            generated["heating_schedule"] = str(outputs_dir / "heating_schedule.pdf")
        except Exception as exc:
            logger.warning("plot_heating_schedule failed during postprocess: %s", exc)

    artifacts_path = outputs_dir / "postprocess_artifacts.json"
    with artifacts_path.open("w") as f:
        json.dump({"result_path": str(result_path), "generated": str(generated)}, f, indent=2, default=str, sort_keys=True)
    return {"result_path": str(result_path), "artifacts_path": str(artifacts_path)}


def _bundle_to_tmtx_shim(bundle):
    """Rebuild the minimum object surface required by TORAX plot helpers."""
    from OpenFUSIONToolkit.TokaMaker import pulse_design as pulse_design_mod

    def _coerce_numeric_payload(value):
        if isinstance(value, dict):
            return {str(k): _coerce_numeric_payload(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            coerced_items = [_coerce_numeric_payload(v) for v in value]
            if not coerced_items:
                return np.asarray([])
            if all(isinstance(v, (int, float, bool, np.number, np.ndarray)) for v in coerced_items):
                return np.asarray(coerced_items)
            return coerced_items
        return value

    state = _coerce_numeric_payload(bundle.get("state", {}))
    tm_times = np.asarray(bundle.get("tm_times", []), dtype=float)
    current_loop = bundle.get("current_loop", 0) or 0
    flattop = bundle.get("flattop", None)
    if flattop is None:
        flattop = np.zeros(len(tm_times), dtype=bool)
    else:
        flattop = np.asarray(flattop, dtype=bool)
    coil_bounds = bundle.get("coil_bounds", {})
    results = _coerce_numeric_payload(bundle.get("results", {}))

    tm = SimpleNamespace(
        coil_sets={name: {"net_turns": 1.0} for name in coil_bounds},
        lim_contours=[],
        settings=SimpleNamespace(mirror_mode=False),
    )

    def _print(*args, **kwargs):
        logger.info(" ".join(str(a) for a in args))

    shim = SimpleNamespace(
        _state=state,
        _tm_times=tm_times,
        _t_init=float(tm_times[0]) if len(tm_times) else 0.0,
        _t_final=float(tm_times[-1]) if len(tm_times) else 0.0,
        _last_surface_factor=float(state.get("last_surface_factor", bundle.get("last_surface_factor", 0.0))),
        _current_loop=current_loop,
        _flattop=flattop,
        _coil_bounds=coil_bounds,
        _results=results,
        _output_mode=bundle.get("output_mode"),
        _tm=tm,
        _print=_print,
        _run_name=Path(bundle.get("actor_checkpoint", "actor_eval")).stem,
    )

    # Bind the notebook-facing helpers directly to the shim so offline
    # postprocessing can emit the same figures/movie/summary without rerunning
    # TORAX.
    shim.summary = lambda **kwargs: pulse_design_mod.summary(shim, **kwargs)
    shim.plot_scalars = lambda save_path=None, display=True, **kwargs: pulse_design_mod.plot_scalars(
        shim, save_path=save_path, display=display, **kwargs
    )
    shim.plot_profile_evolution = lambda save_path=None, display=True, one_plot=False, **kwargs: pulse_design_mod.plot_profile_evolution(
        shim, save_path=save_path, display=display, one_plot=one_plot, **kwargs
    )
    shim.plot_lcfs_evolution = lambda save_path=None, display=True, one_plot=False, **kwargs: pulse_design_mod.plot_lcfs_evolution(
        shim, save_path=save_path, display=display, one_plot=one_plot, **kwargs
    )
    shim.make_movie = lambda save_path=None, **kwargs: pulse_design_mod.make_movie(
        shim, save_path=save_path, **kwargs
    )
    return shim
