r'''! Minimal ITER test for pulse_design.py: the TokaMaker_TORAX coupled pulse design workflow.

Builds two flattop-like seed equilibria (same Ip),
runs a 5 s simulation at constant Ip, and returns a flat dict of scalars.

Expects ITER_mesh.h5 produced from src/examples/TokaMaker/ITER/ITER_mesh_ex.ipynb,
or set environment variable TOKAMAKER_ITER_MESH to the .h5 path.

'''
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import types
from typing import Any, Dict, Optional

import numpy as np
import pytest

_NUMERIC = (int, float, np.integer, np.floating)


def _round_outputs(obj: Any, ndigits: int = 2) -> Any:
    r'''! Round floats (and ints as floats) for JSON-friendly regression output; preserve bool, None, str.'''
    if obj is None or isinstance(obj, bool) or isinstance(obj, str):
        return obj
    if isinstance(obj, _NUMERIC):
        return round(float(obj), ndigits)
    if isinstance(obj, dict):
        return {k: _round_outputs(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_round_outputs(x, ndigits) for x in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj

# Repo root: walk upward from this file until src/python exists (works for
# src/tests/physics/, tests/, etc.).
def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "src", "python")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "Could not find repository root (no parent directory contains src/python). "
                "Set TOKAMAKER_ITER_MESH to the ITER_mesh.h5 path."
            )
        d = parent


_REPO_ROOT = _repo_root()
_PYTHON_SRC = os.path.join(_REPO_ROOT, "src", "python")
if _PYTHON_SRC not in sys.path:
    sys.path.insert(0, _PYTHON_SRC)


def _default_iter_mesh_path() -> str:
    env = os.environ.get("TOKAMAKER_ITER_MESH")
    if env:
        return os.path.abspath(env)
    return os.path.join(
        _REPO_ROOT,
        "src",
        "examples",
        "TokaMaker",
        "ITER",
        "ITER_mesh.h5",
    )


def _array_to_profile_dict(profile_array: np.ndarray, psi_grid: np.ndarray) -> Dict[float, float]:
    return {float(p): float(v) for p, v in zip(psi_grid, profile_array)}


def _parabolic_profile(edge: float, core: float, psi_grid: np.ndarray, alpha: float = 1.8) -> np.ndarray:
    return edge + (core - edge) * (1.0 - np.asarray(psi_grid) ** alpha)


def _build_min_norm_coil_reg(mygs) -> None:
    reg_terms = []
    for name in mygs.coil_sets:
        if name.startswith("CS"):
            weight = 2.0e-2 if name.startswith("CS1") else 1.0e-2
            reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=weight))
        elif name.startswith("PF"):
            reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e-2))
        elif name.startswith("VS"):
            reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e-2))
    reg_terms.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1.0e2))
    mygs.set_coil_reg(reg_terms=reg_terms)


def _final_tmtx_equilibrium_stats(tt) -> Dict[str, Any]:
    r'''! Scalars from each TokaMaker equilibrium produced in the last fly() TM pass.'''
    out: Dict[str, Any] = {}
    out["tmtx_last_completed_loop"] = int(getattr(tt, "_current_loop", -1))
    equil = tt.state.get("equil", {}) or {}
    tm_times = list(tt._tm_times)
    for i in sorted(equil.keys(), key=lambda k: int(k) if isinstance(k, (int, np.integer)) else k):
        eq = equil[i]
        if eq is None:
            continue
        try:
            stats = eq.get_stats(li_normalization="iter")
        except Exception:
            continue
        pfx = f"final_equil_i{i}_"
        out[f"{pfx}time_s"] = float(tm_times[i]) if i < len(tm_times) else float("nan")
        for key, val in stats.items():
            if key == "Ip_centroid":
                out[f"{pfx}Ip_centroid_R_m"] = float(val[0])
                out[f"{pfx}Ip_centroid_Z_m"] = float(val[1])
            elif isinstance(val, (float, int, np.floating, np.integer)):
                out[f"{pfx}{key}"] = float(val)
            elif isinstance(val, np.ndarray):
                out[f"{pfx}{key}"] = val.astype(float).tolist()
            else:
                out[f"{pfx}{key}"] = val
        try:
            out[f"{pfx}diverted"] = bool(eq.diverted)
            out[f"{pfx}psi_lcfs_Wb_per_rad"] = float(eq.psi_bounds[0])
            out[f"{pfx}psi_axis_Wb_per_rad"] = float(eq.psi_bounds[1])
        except Exception:
            pass
        try:
            out[f"{pfx}vloop_V"] = float(eq.calc_loopvoltage())
        except ValueError:
            out[f"{pfx}vloop_V"] = None
    return out


def _iter_baseline_shape() -> tuple[np.ndarray, np.ndarray]:
    r'''! LCFS isoflux points and X-point from ITER_baseline_ex.ipynb (L-mode inverse case).'''
    isoflux_pts = np.array(
        [
            [8.20, 0.41],
            [8.06, 1.46],
            [7.51, 2.62],
            [6.14, 3.78],
            [4.51, 3.02],
            [4.26, 1.33],
            [4.28, 0.08],
            [4.49, -1.34],
            [7.28, -1.89],
            [8.00, -0.68],
        ]
    )
    x_point = np.array([[5.125, -3.4]])
    return isoflux_pts, x_point


def _run_tokamaker_torax(
    mesh_path: Optional[str] = None,
    nthreads: int = 2,
    t_final: float = 5.0,
    tx_dt: float = 0.5,
    Ip_flattop: float = 15.0e6,
    pax_a: float = 6.2e5,
    pax_b: float = 5.7e5,
    eqdsk_nr: int = 100,
    eqdsk_nz: int = 100,
    max_loop: int = 1,
    loop0: bool = True,
) -> Dict[str, Any]:
    r'''! Run the minimal ITER TokaMaker_TORAX benchmark and return numerical outputs.
    
        Parameters
        ----------
        mesh_path
            Path to ITER_mesh.h5. Default: TOKAMAKER_ITER_MESH or
            src/examples/TokaMaker/ITER/ITER_mesh.h5 under the repo root.
        ip_flat
            Constant plasma current [A] for both seeds and the TORAX Ip schedule.
        pax_a, pax_b
            Axis pressure targets [Pa] for the two seeds (slight mismatch so transport
            has something to do while Ip stays flat).
        max_loop
            Highest counted coupling index (see TokaMaker_TORAX.fly); default 1.
        loop0
            If True (default), run the cheap index-0 pass before counted loops.
        
    '''
    try:
        import torax  # noqa: F401
    except ImportError:
        pytest.skip("TokaMaker_TORAX requires the torax package.")

    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX, summary
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, read_eqdsk

    mesh = os.path.abspath(mesh_path or _default_iter_mesh_path())
    if not os.path.isfile(mesh):
        raise FileNotFoundError(
            f"ITER mesh not found at {mesh!r}."
        )

    r0_geo = 6.3
    b0 = 5.2
    z0 = 0.5
    f0 = r0_geo * b0

    isoflux_pts, x_point = _iter_baseline_shape()
    diverted_pts = np.vstack((isoflux_pts, x_point))

    work_dir = tempfile.mkdtemp(prefix="test_tmtx_")
    eq_a = os.path.join(work_dir, "seed_t0.eqdsk")
    eq_b = os.path.join(work_dir, "seed_t5.eqdsk")

    myoft = OFT_env(nthreads=nthreads)
    mygs = TokaMaker(myoft)

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.maxits = 500
    mygs.setup(order=2, F0=f0)
    mygs.set_coil_vsc({"VS": 1.0})

    coil_bounds = {key: [-50.0e6, 50.0e6] for key in mygs.coil_sets}
    mygs.set_coil_bounds(coil_bounds)

    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    seed_metrics: Dict[str, float] = {}

    for idx, (pax, eq_path) in enumerate(((pax_a, eq_a), (pax_b, eq_b))):
        mygs.set_isoflux_constraints(diverted_pts)
        mygs.set_saddle_constraints(x_point)
        mygs.set_targets(Ip=Ip_flattop, pax=pax)
        _build_min_norm_coil_reg(mygs)
        mygs.init_psi()
        mygs.solve()
        seed_metrics[f"seed{idx}_pax_Pa"] = float(pax)
        mygs.save_eqdsk(eq_path, cocos=2, nr=eqdsk_nr, nz=eqdsk_nz)
        g = read_eqdsk(eq_path)
        seed_metrics[f"seed{idx}_psi_lcfs_Wb_per_rad"] = float(-g["psibry"])

    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        tm_times = np.linspace(0.0, t_final, 2)
        tt = TokaMaker_TORAX(
            t_init=0.0,
            t_final=t_final,
            eqtimes=[0.0, t_final],
            g_eqdsk_arr=["seed_t0.eqdsk", "seed_t5.eqdsk"],
            tokamaker_obj=mygs,
            tx_dt=tx_dt,
            tm_times=tm_times,
            last_surface_factor=0.99,
            truncate_eq=False,
        )

        tt.set_TORAX_grid(grid_type="n_rho", grid=51)

        n_sample = 100
        psi_sample = np.linspace(0.0, 1.0, n_sample)
        ne_init = _parabolic_profile(0.25e20, 2.0e20, psi_sample)
        te_init = _parabolic_profile(0.10, 10.0, psi_sample)
        ne = {0.0: _array_to_profile_dict(ne_init, psi_sample)}
        te = {0.0: _array_to_profile_dict(te_init, psi_sample)}
        tt.set_ne(ne, right_bc=0.25e20)
        tt.set_Te(te, right_bc=0.1)
        tt.set_Ti(te, right_bc=0.1)
        tt.set_pedestal(set_pedestal=True, T_i_ped=3.0, T_e_ped=3.0, n_e_ped=0.8e20)

        heat_times = {0.0: 30.0e6}
        tt.set_heating(
            generic_heat=heat_times,
            generic_heat_loc=0.25,
            nbi_current=True,
            ecrh={0.0: 20.0e6},
            ecrh_loc=0.35,
            fusion=True,
            ei_exchange=True,
        )
        tt.set_fueling(gas_puff_S_total=1e22, gas_puff_decay_length=0.05, pellet_deposition_location=0.8, pellet_width=0.1, pellet_S_total={0: 5e21})

        tt.set_Ip({0.0: Ip_flattop})
        tt.set_plasma_composition(Zeff=1.6, main_ion={"D": 0.5, "T": 0.5}, impurity="Ne")
        tt.set_evolve(density=True, Ti=True, Te=True, current=True)

        tt.fly(
            run_name="tmp",
            max_loop=max_loop,
            loop0=loop0,
            output_mode=False,
            initial_relax=True,
        )

        phys = summary(tt)
    finally:
        os.chdir(prev_cwd)

    st = tt.state
    times = np.asarray(tt._tm_times)
    out: Dict[str, Any] = {}
    out.update(seed_metrics)
    out["Ip_flattop_MA"] = float(Ip_flattop / 1.0e6)
    out["t_final_s"] = float(t_final)
    out["tx_dt_s"] = float(tx_dt)
    out["n_tm_times"] = int(len(times))

    # TokaMaker-side traces (good for regression; values evolve with TORAX coupling).
    out["psi_lcfs_tm_first_Wb_per_rad"] = float(st["psi_lcfs_tm"][0])
    out["psi_lcfs_tm_last_Wb_per_rad"] = float(st["psi_lcfs_tm"][-1])
    out["Ip_tm_first_MA"] = float(st["Ip_tm"][0] / 1.0e6)
    out["Ip_tm_last_MA"] = float(st["Ip_tm"][-1] / 1.0e6)
    out["q95_tm_min"] = float(np.min(st["q95_tm"][st["q95_tm"] > 0])) if np.any(st["q95_tm"] > 0) else None
    out["q95_tm_last"] = float(st["q95_tm"][-1])
    out["beta_N_tm_max"] = float(np.max(st["beta_N_tm"]))

    out.update(_final_tmtx_equilibrium_stats(tt))

    # Merge physics summary (TORAX / integrated quantities).
    for k, v in phys.items():
        if v is not None and k not in out:
            out[k] = v

    out["_work_dir"] = work_dir

    # tt.plot_scalars(display=True)
    # tt.plot_profile_evolution(display=True, one_plot=True)
    # tt.plot_lcfs_evolution(display=True, one_plot=True)

    rounded: Dict[str, Any] = {}
    for k, v in out.items():
        if k.startswith("_"):
            rounded[k] = v
        else:
            rounded[k] = _round_outputs(v, ndigits=2)
    return rounded


def test_tokamaker_torax() -> None:
    r'''! Run the minimal ITER TokaMaker_TORAX benchmark and assert key outputs exist.'''
    result = _run_tokamaker_torax()
    assert isinstance(result, dict)
    assert "Ip_flattop_MA" in result
    assert "t_final_s" in result


class _RLLoopVoltage:
    def sel(self, **kwargs):
        return 0.0


class _RLScalars:
    v_loop_lcfs = _RLLoopVoltage()


class _RLDataTree:
    scalars = _RLScalars()


def _minimal_rl_tmtx(pulse_design: Any, events: Optional[list] = None) -> Any:
    tmtx = object.__new__(pulse_design.TokaMaker_TORAX)
    tmtx._rl_actor = object()
    tmtx._rl_actor_checkpoint = None
    tmtx._rl_actions_history = []
    tmtx._rl_event_callback = None if events is None else events.append
    tmtx._rl_max_action_power_w = None
    tmtx._t_final = 100.0
    tmtx._t_init = 0.0
    tmtx._tm_times = np.array([0.0, 100.0])
    tmtx._state = {"psi_lcfs_tx": np.array([1.0, 0.5])}
    tmtx._save_outputs = False
    tmtx._steady_state_mode = False
    tmtx._current_loop = 1
    tmtx._log = lambda message: None
    tmtx._print = lambda message: None
    tmtx._merge_rl_heating_schedules = pulse_design.TokaMaker_TORAX._merge_rl_heating_schedules
    tmtx._extract_rl_state_vector = lambda *args, **kwargs: np.zeros(pulse_design.RL_STATE_DIM)
    tmtx._capture_relax_tx_profiles_from_datatree = lambda *args, **kwargs: None
    tmtx._tx_update = lambda *args, **kwargs: None
    return tmtx


def test_rl_actor_state_dim_mismatch_raises(tmp_path, monkeypatch) -> None:
    r'''! A supplied actor checkpoint with the wrong observation size must fail fast.'''
    torch = pytest.importorskip("torch")

    pulse_design = _load_pulse_design_for_unit(monkeypatch)
    checkpoint_path = tmp_path / "bad_actor.pt"
    torch.save(
        {
            "action_max": torch.ones(2),
            "state_mean": torch.zeros(pulse_design.RL_STATE_DIM + 1),
            "state_std": torch.ones(pulse_design.RL_STATE_DIM + 1),
            "actor": {},
        },
        checkpoint_path,
    )

    tmtx = object.__new__(pulse_design.TokaMaker_TORAX)
    tmtx._rl_actor = None
    tmtx._rl_actor_checkpoint = str(checkpoint_path)
    messages = []
    tmtx._log = messages.append
    tmtx._print = messages.append

    with pytest.raises(ValueError, match=r"Checkpoint state_dim .* != RL_STATE_DIM"):
        tmtx._run_tx_rl_segmented()

    assert not any("baseline heating fallback" in msg for msg in messages)


def test_rl_actor_action_watts_are_not_scaled_twice(monkeypatch) -> None:
    r'''! Actor actions stay in watts internally and are reported in MW.'''
    pulse_design = _load_pulse_design_for_unit(monkeypatch)
    monkeypatch.setattr(pulse_design, "RL_DECISION_TIMES", [80])
    monkeypatch.setattr(pulse_design, "RL_DECISION_T_LAST", 80)

    events = []
    segments = []
    tmtx = _minimal_rl_tmtx(pulse_design, events=events)

    def run_tx_segment(t_start, t_end, ecrh_powers, nbi_powers):
        segments.append((t_start, t_end, dict(ecrh_powers), dict(nbi_powers)))
        return _RLDataTree(), None

    tmtx._run_tx_segment = run_tx_segment
    tmtx._rl_select_action_w = lambda *args, **kwargs: np.array([12.0e6, 34.0e6])

    tmtx._run_tx_rl_segmented()

    expected_event = {
        "event": "decision",
        "decision_index": 0,
        "decision_t": 80.0,
        "knot_t": 100.0,
        "ecrh_W": 12.0e6,
        "nbi_W": 34.0e6,
        "ecrh_MW": 12.0,
        "nbi_MW": 34.0,
    }
    assert events == [expected_event]

    _, _, ecrh_powers, nbi_powers = segments[-1]
    assert ecrh_powers[100.0] == 12.0e6
    assert nbi_powers[100.0] == 34.0e6


def test_rl_actor_action_cap_raises_before_next_segment(monkeypatch) -> None:
    r'''! Unphysical RL actions fail before launching another TORAX segment.'''
    pulse_design = _load_pulse_design_for_unit(monkeypatch)
    monkeypatch.setattr(pulse_design, "RL_DECISION_TIMES", [80])
    monkeypatch.setattr(pulse_design, "RL_DECISION_T_LAST", 80)

    segments = []
    tmtx = _minimal_rl_tmtx(pulse_design)
    tmtx._rl_max_action_power_w = 100.0e6
    tmtx._run_tx_segment = lambda *args, **kwargs: (segments.append(args) or (_RLDataTree(), None))
    tmtx._rl_select_action_w = lambda *args, **kwargs: np.array([101.0e6, 34.0e6])

    with pytest.raises(ValueError, match=r"exceeds RL max action power"):
        tmtx._run_tx_rl_segmented()

    assert len(segments) == 1


def test_rl_segment_timeout_wraps_torax_run(monkeypatch) -> None:
    r'''! RL segment timeout errors include the segment window.'''
    pulse_design = _load_pulse_design_for_unit(monkeypatch)

    class _TimeoutContext:
        def __enter__(self):
            raise TimeoutError("timeout marker")

        def __exit__(self, exc_type, exc, tb):
            return False

    tmtx = object.__new__(pulse_design.TokaMaker_TORAX)
    tmtx._loop0_coarse_tx_main_scope = lambda: _null_context()
    tmtx._get_tx_config_segment = lambda *args, **kwargs: object()
    tmtx._rl_segment_timeout = lambda *args, **kwargs: _TimeoutContext()
    tmtx._print = lambda message: None
    tmtx._log = lambda message: None
    monkeypatch.setattr(
        pulse_design.torax,
        "run_simulation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("timeout wrapper not used")),
    )

    with pytest.raises(TimeoutError, match=r"TORAX RL run \[0, 80\].*timeout"):
        tmtx._run_tx_segment(0.0, 80.0, {}, {})


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _load_pulse_design_for_unit(monkeypatch):
    oft_pkg = types.ModuleType("OpenFUSIONToolkit")
    oft_pkg.__path__ = []
    tokamaker_pkg = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    tokamaker_pkg.__path__ = []
    util_mod = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
    util_mod.read_eqdsk = lambda *args, **kwargs: None
    util_mod.create_power_flux_fun = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", oft_pkg)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tokamaker_pkg)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.util", util_mod)

    module_path = os.path.join(
        _PYTHON_SRC, "OpenFUSIONToolkit", "TokaMaker", "pulse_design.py"
    )
    spec = importlib.util.spec_from_file_location("_test_pulse_design_rl", module_path)
    assert spec is not None and spec.loader is not None
    pulse_design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pulse_design)
    return pulse_design


def main() -> None:
    t_wall0 = time.perf_counter()
    result = _run_tokamaker_torax()
    elapsed_s = time.perf_counter() - t_wall0
    slim = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(slim, indent=2, sort_keys=True))
    print(f"\nTotal wall time (script): {elapsed_s:.2f} s", flush=True)


if __name__ == "__main__":
    main()
