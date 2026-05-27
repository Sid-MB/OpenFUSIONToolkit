"""
collect_trajectories.py
=======================
Generates a static offline RL dataset by running TokaMaker_TORAX simulations
with varied ECRH and NBI heating schedules sampled via Latin Hypercube Sampling.

Dataset structure (per trajectory, 21 transitions):
    s       : state vector at rl_time[i]
    a       : [ecrh_W, nbi_W] applied from rl_time[i] to rl_time[i+1]
    r       : mean Q_fusion from rl_time[i] to rl_time[i+1]
              (plus terminal reward at last step)
    s_next  : state vector at rl_time[i+1]
    done    : True at last transition

Usage:
    python collect_trajectories_delta.py --n_trajectories 500 --output_dir ./rl_dataset
    python collect_trajectories_delta.py --n_trajectories 1000 --start_idx 600 --output_dir ./rl_dataset_delta_sampling_maxloop=2_grid_51

Output layout:
    output_dir/run_manifest.json
    output_dir/all_actions.npy
    output_dir/full_trajectories/trajectory_<run_id>.zarr
    output_dir/trajectories/trajectory_<run_id>.json  # optional, with --save_json
    output_dir/failures/failed_run_<run_id>.json
    output_dir/chunks/<chunk>/task_status.json
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
import signal
import copy
import types
import numpy as np
from scipy.stats import qmc
from datetime import datetime
import io
from contextlib import contextmanager, redirect_stdout
from functools import partial
import shutil
import subprocess

from dataloader import (
    create_run_manifest,
    dataset_paths,
    ensure_dataset_dirs,
    initialize_dataset,
    record_task_status,
    require_dataset,
    save_failure_atomic,
    save_full_trajectory_zarr_atomic,
    save_trajectory_atomic,
    full_trajectory_zarr_path,
    trajectory_path,
)


# ── RL / simulation config ────────────────────────────────────────────────────

# Times at which RL states, actions, and rewards are collected
RL_TIMES = list(range(80, 501, 20))   # [80, 100, 120, ..., 480, 500] — 22 values
DECISION_TIMES = RL_TIMES[:-1]         # [80, 100, ..., 480] — 21 values

# TokaMaker solve times (fine resolution for physics accuracy)
N_TM_POINTS = 10
TM_TIMES = np.linspace(0, 600, N_TM_POINTS)

# Action bounds
ECRH_MIN, ECRH_MAX = 0.0, 40.0E6
NBI_MIN,  NBI_MAX  = 0.0, 33.0E6

# NBI is physically off before L-H transition
NBI_ZERO_BEFORE = 80  # seconds

# Safety thresholds for penalty
# source: https://www.iter.org/sites/default/files/education/L02_Wagner.pdf
# source: https://www.osti.gov/servlets/purl/6227385
Q95_MIN     = 3.0
BETA_N_MAX  = 2.8
FGW_MAX     = 0.85

# Terminal reward weight on flux consumption
FLUX_WEIGHT = 0.001  # R_terminal = Q_flattop_avg - FLUX_WEIGHT * flux_consumed_Wb
Q95_PENALTY_WEIGHT = 0.15
BETA_N_PENALTY_WEIGHT = 1.67
FGW_PENALTY_WEIGHT = 3

# Pellet schedule (fixed to baseline)
PELLET_S_TOTAL = {0: 0, 90: 5e21, 450: 5e21, 451: 0}
SEED_EQDSK_COUNT = 5


# ── Per-worker global state ───────────────────────────────────────────────────

_mygs = None

FATAL_EXCEPTIONS = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    ModuleNotFoundError,
    ImportError,
    OSError,
)


def require_readable_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{description} is missing: {path}')
    if not os.access(path, os.R_OK):
        raise PermissionError(f'{description} is not readable: {path}')
    if os.path.getsize(path) == 0:
        raise OSError(f'{description} is empty: {path}')


def resolve_seed_eqdsk_paths(cwd):
    """
    Resolve the five seed EQDSKs without silently falling back on partial data.

    The current layout keeps seeds in seed_eqdsks/. A legacy root-level layout is
    still accepted only when all five files are present there.
    """
    candidates = [
        os.path.join(cwd, 'seed_eqdsks'),
        cwd,
    ]
    checked = []

    for directory in candidates:
        paths = [os.path.join(directory, f'i={i}.eqdsk') for i in range(SEED_EQDSK_COUNT)]
        existing = [path for path in paths if os.path.isfile(path)]
        checked.extend(paths)
        if len(existing) == SEED_EQDSK_COUNT:
            return paths
        if existing:
            missing = [path for path in paths if not os.path.isfile(path)]
            raise FileNotFoundError(
                'Seed EQDSK directory is incomplete. Missing files:\n'
                + '\n'.join(f'  {path}' for path in missing)
            )

    raise FileNotFoundError(
        'Seed EQDSK files were not found. Checked:\n'
        + '\n'.join(f'  {path}' for path in checked)
    )


def preflight_required_inputs(cwd, eqdsk_list, initial_relax_cache=None,
                              require_initial_relax_cache=False):
    require_readable_file(os.path.join(cwd, 'ITER_mesh.h5'), 'TokaMaker mesh')
    for i, eqdsk_path in enumerate(eqdsk_list):
        require_readable_file(eqdsk_path, f'seed EQDSK i={i}')
    if require_initial_relax_cache:
        require_readable_file(initial_relax_cache, 'initial relax cache')


class TrajectoryTimeoutError(RuntimeError):
    """Raised when a single trajectory exceeds its configured wall-time budget."""


@contextmanager
def trajectory_timeout(seconds):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise TrajectoryTimeoutError(
            f'trajectory exceeded timeout of {seconds} seconds'
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def nvidia_gpu_visible():
    """Return True when this process appears to have an NVIDIA GPU available."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        return False

    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, '-L'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return False

    return result.returncode == 0 and 'GPU ' in result.stdout


def validate_jax_backend(require_cuda_on_gpu=True):
    """
    Fail fast on GPU nodes if JAX only initialized a CPU backend.

    TORAX uses JAX. Without a CUDA-enabled jaxlib/plugin installation, a Slurm
    GPU allocation can silently run TORAX on CPU unless we stop here.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        print('JAX backend check skipped: CUDA_VISIBLE_DEVICES=-1')
        return 'cpu-forced', []

    try:
        import jax
    except Exception as e:
        raise RuntimeError(f'JAX import failed: {e}') from e

    backend = jax.default_backend()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == 'gpu']
    gpu_visible = nvidia_gpu_visible()

    print(f'JAX backend: {backend}; devices: {devices}')

    if require_cuda_on_gpu and gpu_visible and not gpu_devices:
        raise RuntimeError(
            'An NVIDIA GPU is visible, but JAX did not initialize any GPU '
            'devices. Install/run with CUDA-enabled JAX, e.g. '
            '`uv run --extra cuda13 ...`, '
            'or pass --allow_cpu_jax_on_gpu to override.'
        )

    return backend, devices


def worker_init(cwd, require_cuda_on_gpu):
    """Initialize one OFT/TokaMaker object per worker process."""
    global _mygs
    os.chdir(cwd)
    validate_jax_backend(require_cuda_on_gpu=require_cuda_on_gpu)
    _mygs, _, _, _ = setup_tokamaker(cwd)


# ── Latin Hypercube Sampling ──────────────────────────────────────────────────

def sample_actions_lhs(n_trajectories, seed=42):
    """
    LHS samples delta (change per step) rather than absolute values.
    This enforces smoothness — large jumps between steps are impossible
    because the delta itself is bounded.

    max_delta controls the maximum change per 20s step in watts.
    """
    ECRH_DEFAULT = 20.0e6  # starting value at t=100 (end of fixed ramp-up)
    NBI_DEFAULT  = 33.0e6

    # Max change per step — tune these to control smoothness
    # 2 MW per 20s step means at most 40 MW total swing over flattop
    ECRH_DELTA_MAX = 2.0e6
    NBI_DELTA_MAX  = 2.0e6

    n_decision = len(DECISION_TIMES)  # 21
    n_params = n_decision * 2

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_trajectories)

    # Scale to [-delta_max, +delta_max]
    lower = np.array([-ECRH_DELTA_MAX, -NBI_DELTA_MAX] * n_decision)
    upper = np.array([ ECRH_DELTA_MAX,  NBI_DELTA_MAX] * n_decision)
    deltas = qmc.scale(samples, lower, upper)
    deltas = deltas.reshape(n_trajectories, n_decision, 2)

    # Cumsum from the default starting point
    actions = np.zeros_like(deltas)
    for i in range(n_trajectories):
        ecrh = ECRH_DEFAULT + np.cumsum(deltas[i, :, 0])
        nbi  = NBI_DEFAULT  + np.cumsum(deltas[i, :, 1])

        # Clip to physical bounds
        actions[i, :, 0] = np.clip(ecrh, ECRH_MIN, ECRH_MAX)
        actions[i, :, 1] = np.clip(nbi,  NBI_MIN,  NBI_MAX)

    return actions


# ── Schedule builders ─────────────────────────────────────────────────────────
def build_ecrh_schedule(action_row):
    schedule = {}

    # Fixed ramp-up (not agent-controlled)
    schedule[0]  = 0
    schedule[10] = 0
    schedule[50] = 20.0E6
    schedule[80] = 20.0E6

    # Agent-controlled: action at decision time t sets heating for interval t→t+20
    # So we write the value at t+20 (when TokaMaker picks it up)
    for i, t in enumerate(DECISION_TIMES):
        t_apply = t + 20   # 100, 120, ..., 500
        schedule[t_apply] = float(action_row[i, 0])

    # Fixed ramp-down
    schedule[520] = 15.0E6
    schedule[540] = 10.0E6
    schedule[560] = 5.0E6
    schedule[600] = 0.0

    return schedule


def build_nbi_schedule(action_row):
    """
    Build generic_powers dict from action_row (n_decision_times, 2).
    NBI is 0 before t=80 by construction.
    Fixed at 33 MW at t=80, agent-controlled from t=100 to t=500,
    fixed ramp-down from t=520 onward.
    action_row[:, 1] = nbi_W at each decision time.
    """
    schedule = {}

    # Fixed ramp-up (NBI off before L-H transition)
    schedule[0]  = 0.0
    schedule[79] = 0.0
    schedule[80] = 33.0E6

    # Agent-controlled: decision at t applies at t+20
    for i, t in enumerate(DECISION_TIMES):
        t_apply = t + 20  # 100, 120, ..., 500
        schedule[t_apply] = float(action_row[i, 1])

    # Fixed ramp-down
    schedule[520] = 10.0E6
    schedule[540] = 7.0E6
    schedule[560] = 4.0E6
    schedule[580] = 2.0E6
    schedule[600] = 0.0

    return schedule


# ── State extraction ──────────────────────────────────────────────────────────

def get_nearest_tm_index(rl_time, tm_times):
    """Return index of closest TM timepoint to rl_time."""
    return int(np.argmin(np.abs(np.array(tm_times) - rl_time)))


def interpolate_torax_scalar(tmtx, var_name, rl_time):
    """Interpolate a TORAX DataTree scalar to a specific rl_time."""
    try:
        torax_times = tmtx._data_tree['scalars'].coords['time'].values
        values = tmtx._data_tree['scalars'][var_name].values.astype(float)
        return float(np.interp(rl_time, torax_times, values))
    except Exception as e:
        raise RuntimeError(
            f'Failed to interpolate TORAX scalar {var_name!r} at t={rl_time}.'
        ) from e


def get_torax_profile_at_rho(tmtx, var_name, rho_values, rl_time):
    """
    Interpolate a TORAX profile variable to specific rho points at rl_time.
    Returns list of floats, one per rho value.
    """
    ds = tmtx._data_tree['profiles']
    torax_times = ds.coords['time'].values

    # Find nearest time index in TORAX
    t_idx = int(np.argmin(np.abs(torax_times - rl_time)))
    profile_data = ds[var_name].values  # shape (time, rho)

    # Check which rho coordinate this variable uses.
    var_dims = ds[var_name].dims

    if 'rho_face_norm' in var_dims:
        rho_coord = ds.coords['rho_face_norm'].values
    elif 'rho_norm' in var_dims:
        rho_coord = ds.coords['rho_norm'].values
    elif 'rho_cell_norm' in var_dims:
        rho_coord = ds.coords['rho_cell_norm'].values
    else:
        raise RuntimeError(
            f'No rho coordinate found for TORAX profile {var_name!r}: dims={var_dims}.'
        )

    profile_at_t = profile_data[t_idx, :]
    result = [float(np.interp(rho, rho_coord, profile_at_t)) for rho in rho_values]
    return result


def extract_state(tmtx, t_start, t_end, current_action):
    """
    Extract state vector for interval [t_start, t_end].

    Returns dict with all state quantities. Scalars are taken at t_start.
    Q_fusion is the mean over the interval [t_start, t_end].
    """
    rho_points = [0.2, 0.5, 0.8]
    state = {}

    # ── Scalars from TORAX DataTree at t_start ────────────────────────────────
    torax_scalars = [
        'H98',
        'tau_E',
        'W_thermal_total',
        'P_SOL_total',
        'P_radiation_e',
        'P_aux_total',
        'f_non_inductive',
        'n_e_line_avg',
        'fgw_n_e_line_avg',
        'T_e_volume_avg',
        'T_i_volume_avg',
        'n_e_volume_avg',
        'beta_N',
        'li3',
        'dW_thermal_dt_smoothed',
        'P_ohmic_e',
        'q_min',
        'rho_q_min',
        'f_bootstrap',
        'P_alpha_total',
        'q95',
        'v_loop_lcfs',
        'Ip',
        'Q_fusion'
    ]
    for var in torax_scalars:
        state[f'tx_{var}'] = interpolate_torax_scalar(tmtx, var, t_start)

    state['ecrh'] = float(current_action[0])
    state['nbi']  = float(current_action[1])

    # ── P_LH margin ───────────────────────────────────────────────────────────
    P_heat_total = interpolate_torax_scalar(tmtx, 'P_heat_total', t_start)
    P_LH         = interpolate_torax_scalar(tmtx, 'P_LH', t_start)
    state['P_LH_margin'] = P_heat_total / P_LH

    # ── Peaking factors ───────────────────────────────────────────────────────
    T_e_core = get_torax_profile_at_rho(tmtx, 'T_e', [0.0], t_start)[0]
    T_i_core = get_torax_profile_at_rho(tmtx, 'T_i', [0.0], t_start)[0]
    n_e_core = get_torax_profile_at_rho(tmtx, 'n_e', [0.0], t_start)[0]

    state['T_e_peaking'] = T_e_core / state['tx_T_e_volume_avg'] if state['tx_T_e_volume_avg'] > 0 else float('nan')
    state['T_i_peaking'] = T_i_core / state['tx_T_i_volume_avg'] if state['tx_T_i_volume_avg'] > 0 else float('nan')
    state['n_e_peaking'] = n_e_core / state['tx_n_e_volume_avg'] if state['tx_n_e_volume_avg'] > 0 else float('nan')

    torax_times = tmtx._data_tree['scalars'].coords['time'].values
    mask = (torax_times >= t_start) & (torax_times <= t_end)

    # ── Q fusion reward over [t_start, t_end] ────────────────────────────
    if not np.any(mask):
        step_reward = 0.0
    else:
        Q_vals = tmtx._data_tree['scalars']['Q_fusion'].values[mask]
        step_reward = np.log(float(np.nanmean(Q_vals)) + 1)

    state['step_reward'] = step_reward

    # ── Safety penalties over [t_start, t_end] ────────────────────────────
    try:
        q95_vals   = tmtx._data_tree['scalars']['q95'].values[mask]
        betaN_vals = tmtx._data_tree['scalars']['beta_N'].values[mask]
        fgw_vals   = tmtx._data_tree['scalars']['fgw_n_e_line_avg'].values[mask]

        state['penalty_q95']   = float(np.sum(np.maximum(Q95_MIN - q95_vals, 0)))
        state['penalty_betaN'] = float(np.sum(np.maximum(betaN_vals - BETA_N_MAX, 0)))
        state['penalty_fgw']   = float(np.sum(np.maximum(fgw_vals - FGW_MAX, 0)))
    except Exception as e:
        raise RuntimeError(
            f'Failed to compute safety penalties for interval [{t_start}, {t_end}].'
        ) from e

    # ── Profiles at rho = 0.2, 0.5, 0.8 at t_start ────────────────────────────
    profile_vars = ['T_e', 'T_i', 'n_e', 'q', 'magnetic_shear']
    for var in profile_vars:
        vals = get_torax_profile_at_rho(tmtx, var, rho_points, t_start)
        for rho, val in zip(rho_points, vals):
            state[f'{var}_rho{int(rho*10)}'] = val

    return state


# ── Reward computation ────────────────────────────────────────────────────────

def compute_reward(tmtx, t_start, t_end, is_terminal=False):
    """
    Compute reward for the interval [t_start, t_end].

    Step reward: mean Q_fusion over the interval.
    Safety penalty: applied if any safety threshold violated in interval.
    Terminal bonus (last step only): Q_flattop_avg - FLUX_WEIGHT * flux_consumed.
    """
    torax_times = tmtx._data_tree['scalars'].coords['time'].values

    # Mask for this interval
    mask = (torax_times >= t_start) & (torax_times <= t_end)
    if not np.any(mask):
        step_reward = 0.0
    else:
        Q_vals = tmtx._data_tree['scalars']['Q_fusion'].values[mask]
        step_reward = np.log(float(np.nanmean(Q_vals)) + 1)

    # Safety penalties over this interval
    penalty = 0.0

    if t_start >= 80:
        try:
            q95_vals   = tmtx._data_tree['scalars']['q95'].values[mask]
            betaN_vals = tmtx._data_tree['scalars']['beta_N'].values[mask]
            fgw_vals   = tmtx._data_tree['scalars']['fgw_n_e_line_avg'].values[mask]

            penalty += Q95_PENALTY_WEIGHT * np.sum(np.maximum(Q95_MIN - q95_vals, 0))
            penalty += BETA_N_PENALTY_WEIGHT * np.sum(np.maximum(betaN_vals - BETA_N_MAX, 0))
            penalty += FGW_PENALTY_WEIGHT * np.sum(np.maximum(fgw_vals - FGW_MAX, 0))
        except Exception as e:
            raise RuntimeError(
                f'Failed to compute reward safety penalties for interval '
                f'[{t_start}, {t_end}].'
            ) from e

    reward = step_reward - penalty

    # Terminal bonus
    if is_terminal:
        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()

        Q_avg   = summary.get('Q_flattop_avg', 0.0)
        flux_wb = summary.get('flux_consumed_Wb', 0.0)
        reward += Q_avg - FLUX_WEIGHT * flux_wb

    return reward


def iter_numeric_values(value, path='value'):
    """Yield (path, value) for numeric leaves in nested lists/dicts."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_numeric_values(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from iter_numeric_values(item, f'{path}[{idx}]')
    elif isinstance(value, (int, float, np.integer, np.floating)):
        yield path, float(value)


def assert_all_numeric_finite(value, label):
    for path, number in iter_numeric_values(value, label):
        if not np.isfinite(number):
            raise ValueError(f'Non-finite numeric value at {path}: {number}')


def validate_trajectory(transitions, summary, action_row):
    if len(transitions) != len(DECISION_TIMES):
        raise ValueError(
            f'Expected {len(DECISION_TIMES)} transitions, got {len(transitions)}.'
        )
    if action_row.shape != (len(DECISION_TIMES), 2):
        raise ValueError(
            f'Expected action shape {(len(DECISION_TIMES), 2)}, got {action_row.shape}.'
        )

    for idx, transition in enumerate(transitions):
        required = {'s', 'a', 'r', 's_next', 'done', 't', 't_next'}
        missing = required.difference(transition)
        if missing:
            raise ValueError(f'Transition {idx} missing keys: {sorted(missing)}')
        assert_all_numeric_finite(transition['s'], f'transitions[{idx}].s')
        assert_all_numeric_finite(transition['s_next'], f'transitions[{idx}].s_next')
        assert_all_numeric_finite(transition['a'], f'transitions[{idx}].a')
        assert_all_numeric_finite(transition['r'], f'transitions[{idx}].r')

    if not isinstance(summary, dict) or not summary:
        raise ValueError('Trajectory summary is missing or empty.')
    assert_all_numeric_finite(summary, 'summary')


# ── Trajectory builder ────────────────────────────────────────────────────────

def build_trajectory(tmtx, action_row):
    """
    After fly() has completed, extract the full list of transitions.

    Returns list of dicts, each with keys: s, a, r, s_next, done.
    Length = len(DECISION_TIMES) = 21.
    """
    transitions = []

    for i, t in enumerate(DECISION_TIMES):

        t_next = RL_TIMES[i + 1]

        is_terminal = (i == len(DECISION_TIMES) - 1)
        a  = action_row[i].tolist()  # [ecrh_W, nbi_W]
        s = extract_state(tmtx, t, t_next, a)
        r = compute_reward(tmtx, t, t_next, is_terminal=is_terminal)

        transitions.append({
            's':      s,
            'a':      a,          # [ecrh_W, nbi_W]
            'r':      r,
            'done':   is_terminal,
            't':      t,
            't_next': t_next,
        })

    for i, transition in enumerate(transitions):
        if i < len(transitions) - 1:
            transition['s_next'] = copy.deepcopy(transitions[i + 1]['s'])
        else:
            transition['s_next'] = {key: 0.0 for key in transition['s']}

    return transitions


# ── Main simulation setup ─────────────────────────────────────────────────────

def setup_tokamaker(cwd):
    """Initialize OFT and TokaMaker, produce seed eqdsks. Run once per process."""

    import sys

    oft_install = os.environ.get('OFT_SELECTED_INSTALL')
    if oft_install is None:
        oft_root = os.path.abspath(os.path.join(cwd, '../../../../../../'))
        oft_install = os.path.join(oft_root, 'install_release')
    oft_python = os.path.join(oft_install, 'python')
    if os.path.isdir(oft_python) and oft_python not in sys.path:
        sys.path.append(oft_python)

    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux
    import numpy as np

    R0, B0, Z0 = 6.3, 5.2, 0.5

    oft_nthreads = int(os.environ.get('OFT_NUM_THREADS', '1'))
    myOFT = OFT_env(nthreads=oft_nthreads)
    mygs  = TokaMaker(myOFT)

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh('ITER_mesh.h5')
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.maxits = 100
    mygs.setup(order=2, F0=R0 * B0)
    mygs.set_coil_vsc({'VS': 1.0})

    return mygs, R0, B0, Z0


def configure_tmtx(mygs, action_row, eqdsk_list, eqtimes, coil_bounds, x_points,
                   Ip_targets, ne_init, Te_init, psi_sample, grid_size=51):
    """Configure one TokaMaker_TORAX object for a trajectory."""
    from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX
    import numpy as np

    ecrh_schedule = build_ecrh_schedule(action_row)
    nbi_schedule  = build_nbi_schedule(action_row)

    tmtx = TokaMaker_TORAX(
        t_init=0,
        t_final=600,
        tx_dt=5,
        eqtimes=eqtimes,
        g_eqdsk_arr=eqdsk_list,
        last_surface_factor=0.99,
        tm_times=TM_TIMES,
        tokamaker_obj=mygs,
    )

    tmtx.set_TORAX_grid(grid_type='n_rho', grid=grid_size)

    tmtx.set_heating(
        generic_heat=nbi_schedule,
        generic_heat_loc=0.25,
        nbi_current=True,
        ecrh=ecrh_schedule,
        ecrh_loc=0.35,
    )

    tmtx.set_fueling(
        gas_puff_S_total=1e22,
        gas_puff_decay_length=0.05,
        pellet_deposition_location=0.8,
        pellet_width=0.1,
        pellet_S_total=PELLET_S_TOTAL,
    )

    def array_to_profile_dict(arr, grid):
        return {float(p): float(v) for p, v in zip(grid, arr)}

    ne = {0.0: array_to_profile_dict(ne_init, psi_sample)}
    Te = {0.0: array_to_profile_dict(Te_init, psi_sample)}

    tmtx.set_ne(ne, right_bc={0: ne_init[-1], 80: 2e19, 500: 2e19, 600: 0.5e19})
    tmtx.set_Te(Te, right_bc=0.1)
    tmtx.set_Ti(Te, right_bc=0.1)

    ne_ped_val, Te_ped_val = 0.9e20, 3.0
    ped_toggle = {0: False, 79: False, 80: True, 500: True, 501: False, 600: False}
    T_ped  = {80: 1.0, 82: 2.0, 90: Te_ped_val, 500: Te_ped_val, 540: 1.0, 580: 1.0, 600: 1.0}
    n_e_ped = {80: 3e19, 82: ne_ped_val / 2, 90: ne_ped_val, 500: ne_ped_val}
    tmtx.set_pedestal(set_pedestal=ped_toggle, T_i_ped=T_ped, T_e_ped=T_ped,
                      n_e_ped=n_e_ped, ped_top=0.9)

    tmtx.set_Ip({0: Ip_targets[0], 100: Ip_targets[2], 500: Ip_targets[2], 600: Ip_targets[0]})
    tmtx.set_plasma_composition(main_ion={'D': 0.5, 'T': 0.5}, impurity='Ne', Zeff=1.6)
    tmtx.set_evolve(density=True, Ti=True, Te=True, current=True)
    tmtx.set_x_points(diverted_times=(80, 500), x_point_targets=x_points, x_point_weight=100)
    tmtx.set_TokaMaker_coil_reg(coil_bounds=coil_bounds, updownsym=False)

    return tmtx


def build_initial_relax_cache(mygs, action_row, cache_path, eqdsk_list, eqtimes,
                              coil_bounds, x_points, Ip_targets, ne_init,
                              Te_init, psi_sample, log_dir=None, grid_size=51):
    """Run the shared initial TORAX relax once and save it for all trajectories."""
    if os.path.exists(cache_path):
        print(f'Using existing initial relax cache: {cache_path}')
        return cache_path

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    print(f'Building shared initial relax cache: {cache_path}')
    tmtx = configure_tmtx(
        mygs, action_row, eqdsk_list, eqtimes, coil_bounds, x_points,
        Ip_targets, ne_init, Te_init, psi_sample, grid_size=grid_size,
    )
    tmtx.fly(
        output_mode=False,
        max_loop=0,
        run_name='initial_relax_cache',
        initial_relax=True,
        relax=False,
        relax_duration=5,
        save_initial_relax_state=cache_path,
        log_dir=log_dir,
    )
    return cache_path


def normalize_loaded_initial_relax_state(tmtx):
    """Restore tuple-wrapped TORAX profile inputs after JSON cache loading."""
    for attr in ('_psi_init', '_n_e_init', '_T_e_init', '_T_i_init'):
        value = getattr(tmtx, attr, None)
        if isinstance(value, list) and len(value) in (2, 3):
            setattr(tmtx, attr, tuple(value))


def patch_initial_relax_cache_loader(tmtx):
    """Make TokaMaker_TORAX JSON cache loading compatible with current TORAX."""
    original_load = tmtx.load_initial_relax_state

    def load_and_normalize(self, filename):
        result = original_load(filename)
        normalize_loaded_initial_relax_state(self)
        return result

    tmtx.load_initial_relax_state = types.MethodType(load_and_normalize, tmtx)


def run_single_trajectory(mygs, action_row, run_id, cwd, eqdsk_list, eqtimes,
                           coil_bounds, x_points, diverted_isoflux_pts,
                           Ip_targets, ne_init, Te_init, psi_sample,
                           initial_relax_cache=None, log_dir=None,
                           max_loop=2, grid_size=51,
                           trajectory_timeout_seconds=0):
    """
    Configure and run one TokaMaker_TORAX simulation with the given action_row.
    Returns (transitions, summary, data_tree), or (None, None, None) if simulation failed.
    """
    try:
        with trajectory_timeout(trajectory_timeout_seconds):
            tmtx = configure_tmtx(
                mygs, action_row, eqdsk_list, eqtimes, coil_bounds, x_points,
                Ip_targets, ne_init, Te_init, psi_sample, grid_size=grid_size,
            )
            if initial_relax_cache is not None:
                patch_initial_relax_cache_loader(tmtx)

            tmtx.fly(
                output_mode=False,
                max_loop=max_loop,
                run_name=f'tmp_{run_id}',
                t_ave_toggle='flattop',
                t_ave_window=25,
                relax=True,
                relax_duration=5,
                initial_relax_state=initial_relax_cache,
                log_dir=log_dir,
            )

        transitions = build_trajectory(tmtx, action_row)
        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()
        validate_trajectory(transitions, summary, action_row)

        return transitions, summary, tmtx._data_tree

    except FATAL_EXCEPTIONS as e:
        print(f'  [run {run_id}] FATAL: {type(e).__name__}: {e}')
        raise
    except Exception as e:
        print(f'  [run {run_id}] FAILED: {e}')
        return None, None, None


# ── Dataset saving ────────────────────────────────────────────────────────────

def action_manifest_fields():
    return {
        'ecrh_W': [ECRH_MIN, ECRH_MAX],
        'nbi_W': [NBI_MIN, NBI_MAX],
        'nbi_zero_before_s': NBI_ZERO_BEFORE,
    }


def sampler_manifest_fields():
    return {
        'name': 'latin_hypercube_delta',
        'ecrh_default_W': 20.0e6,
        'nbi_default_W': 33.0e6,
        'ecrh_delta_max_W_per_step': 2.0e6,
        'nbi_delta_max_W_per_step': 2.0e6,
        'n_decision': len(DECISION_TIMES),
    }


def trajectory_payload(transitions, summary, action_row, run_id):
    payload = {
        'run_id':      run_id,
        'timestamp':   datetime.now().isoformat(),
        'actions_raw': action_row.tolist(),   # (21, 2) array in W
        'transitions': transitions,            # list of 21 dicts
        'summary':     summary,
    }
    return payload


def save_trajectory(transitions, summary, action_row, run_id, dataset_dir):
    """Save one trajectory as an atomically published JSON file."""
    payload = trajectory_payload(transitions, summary, action_row, run_id)
    return save_trajectory_atomic(dataset_dir, payload)


def save_trajectory_outputs(transitions, summary, action_row, run_id, dataset_dir,
                            data_tree=None, save_full_zarr=True, save_json=False):
    payload = trajectory_payload(transitions, summary, action_row, run_id)
    json_path = None
    if save_json:
        json_path = save_trajectory_atomic(dataset_dir, payload)

    zarr_path = None
    if save_full_zarr:
        if data_tree is None:
            raise ValueError('save_full_zarr=True requires a TORAX data_tree')
        zarr_path = save_full_trajectory_zarr_atomic(
            dataset_dir,
            payload,
            data_tree,
            json_path=json_path,
        )

    paths = [str(path) for path in (json_path, zarr_path) if path is not None]
    if not paths:
        raise ValueError('At least one trajectory output format must be enabled')
    return ' | '.join(paths)


def worker_fn(run_id, all_actions, eqdsk_list, eqtimes, coil_bounds,
              x_points, diverted_isoflux_pts, Ip_targets, ne_init, Te_init,
              psi_sample, dataset_dir, initial_relax_cache, log_dir,
              max_loop, grid_size, trajectory_timeout_seconds, save_full_zarr,
              save_json):
    """Run and save one trajectory using this worker's initialized TokaMaker."""
    global _mygs

    try:
        if save_json:
            existing_path = trajectory_path(dataset_dir, run_id)
            if existing_path.exists():
                raise FileExistsError(
                    f'trajectory output already exists before run starts: {existing_path}'
                )
        if save_full_zarr:
            existing_zarr_path = full_trajectory_zarr_path(dataset_dir, run_id)
            if existing_zarr_path.exists():
                raise FileExistsError(
                    f'full trajectory output already exists before run starts: {existing_zarr_path}'
                )

        action_row = all_actions[run_id]
        cwd = os.getcwd()

        transitions, summary, data_tree = run_single_trajectory(
            _mygs, action_row, run_id, cwd, eqdsk_list, eqtimes,
            coil_bounds, x_points, diverted_isoflux_pts,
            Ip_targets, ne_init, Te_init, psi_sample,
            initial_relax_cache=initial_relax_cache,
            log_dir=log_dir,
            max_loop=max_loop,
            grid_size=grid_size,
            trajectory_timeout_seconds=trajectory_timeout_seconds,
        )

        if transitions is not None:
            path = save_trajectory_outputs(
                transitions, summary, action_row, run_id, dataset_dir,
                data_tree=data_tree,
                save_full_zarr=save_full_zarr,
                save_json=save_json,
            )
            return run_id, True, path

        return run_id, False, 'simulation returned None'

    except FATAL_EXCEPTIONS:
        raise
    except Exception as e:
        return run_id, False, str(e)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajectories', type=int, default=500)
    parser.add_argument('--output_dir',     type=str, default='./rl_dataset')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--n_workers',      type=int, default=1,
                        help='Number of parallel worker processes')
    parser.add_argument('--start_idx',      type=int, default=0,
                        help='Resume from this trajectory index')
    parser.add_argument('--end_idx',        type=int, default=None,
                        help='Exclusive end trajectory index; defaults to n_trajectories')
    parser.add_argument('--initial_relax_cache', type=str, default=None,
                        help='Path for shared initial TORAX relax cache; defaults inside output_dir')
    parser.add_argument('--no_initial_relax_cache', action='store_true',
                        help='Disable shared initial relax cache and run initial relax per trajectory')
    parser.add_argument('--build_initial_relax_cache_only', action='store_true',
                        help='Build the shared initial relax cache and exit without running trajectories')
    parser.add_argument('--init_dataset_only', action='store_true',
                        help='Initialize run_manifest.json/all_actions.npy and exit')
    parser.add_argument('--require_existing_dataset', action='store_true',
                        help='Require run_manifest.json/all_actions.npy to already exist and match')
    parser.add_argument('--chunk_dir', type=str, default=None,
                        help='Per-task directory for logs/status; output_dir remains the shared dataset root')
    parser.add_argument('--allow_cpu_jax_on_gpu', action='store_true',
                        help='Do not fail when an NVIDIA GPU is visible but JAX only sees CPU')
    parser.add_argument('--max_loop', type=int, default=2,
                        help='Maximum TokaMaker/TORAX coupling loop count per trajectory')
    parser.add_argument('--grid_size', type=int, default=51,
                        help='TORAX radial grid size passed to set_TORAX_grid')
    parser.add_argument('--trajectory_timeout_seconds', type=int, default=0,
                        help='Abort a single trajectory after this many seconds; 0 disables timeout')
    parser.add_argument('--save_full_zarr', dest='save_full_zarr', action='store_true',
                        default=True,
                        help='Save full TORAX scalars/profiles and reward components as one Zarr store per trajectory (default)')
    parser.add_argument('--no_save_full_zarr', dest='save_full_zarr', action='store_false',
                        help='Disable per-trajectory full Zarr output')
    parser.add_argument('--save_json', dest='save_json', action='store_true',
                        default=False,
                        help='Also save compact trajectory JSON files; disabled by default')
    parser.add_argument('--no_save_json', dest='save_json', action='store_false',
                        help='Disable compact trajectory JSON output (default)')
    args = parser.parse_args()

    if args.n_workers < 1:
        raise ValueError('--n_workers must be >= 1')
    if not args.save_full_zarr and not args.save_json:
        raise ValueError('At least one output format must be enabled')

    cwd = os.getcwd()
    require_cuda_on_gpu = not args.allow_cpu_jax_on_gpu

    paths = ensure_dataset_dirs(args.output_dir)
    chunk_dir = None
    if args.chunk_dir is not None:
        chunk_dir = os.path.abspath(args.chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

    if chunk_dir is not None:
        log_dir = os.path.abspath(os.path.join(chunk_dir, 'tokamaker_torax_logs'))
    else:
        log_dir = os.path.abspath(os.path.join(args.output_dir, 'tokamaker_torax_logs'))
    os.makedirs(log_dir, exist_ok=True)

    print(f'Sampling {args.n_trajectories} trajectories with LHS (seed={args.seed})')
    expected_actions = sample_actions_lhs(args.n_trajectories, seed=args.seed)
    expected_manifest = create_run_manifest(
        n_trajectories=args.n_trajectories,
        seed=args.seed,
        max_loop=args.max_loop,
        grid_size=args.grid_size,
        decision_times=DECISION_TIMES,
        rl_times=RL_TIMES,
        action_bounds=action_manifest_fields(),
        sampler=sampler_manifest_fields(),
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    if args.require_existing_dataset:
        _, all_actions = require_dataset(
            args.output_dir,
            expected_manifest,
            expected_actions=expected_actions,
        )
    else:
        initialize_dataset(args.output_dir, expected_manifest, expected_actions)
        all_actions = expected_actions
    print(f'Action matrix ready: {paths["actions"]}; shape {all_actions.shape}')

    # ── Fixed simulation parameters ───────────────────────────────────────────
    coil_bounds = {key: [-50.e6, 50.e6] for key in [
        'CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L',
        'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6', 'VS'
    ]}

    Ip_targets = [1.5e6, 5e6, 15e6, 15e6, 1.5e6]
    eqdsk_list = resolve_seed_eqdsk_paths(cwd)
    eqtimes    = [0, 30, 80, 500, 600]
    x_points   = np.array([[5.125, -3.4]])
    diverted_isoflux_pts = np.array([
        [8.20, 0.41], [8.06, 1.46], [7.51, 2.62], [6.14, 3.78],
        [5.10, 3.72], [4.51, 3.02], [4.26, 1.33], [4.28, 0.08],
        [4.49, -1.34], [7.28, -1.89], [8.00, -0.68]
    ])
    psi_sample = np.linspace(0.0, 1.0, 25)
    ne_init = np.array([3.00e+19, 2.73e+19, 2.49e+19, 2.28e+19, 2.09e+19,
                        1.92e+19, 1.78e+19, 1.65e+19, 1.54e+19, 1.44e+19,
                        1.35e+19, 1.27e+19, 1.20e+19, 1.14e+19, 1.09e+19,
                        1.04e+19, 9.98e+18, 9.61e+18, 9.29e+18, 9.00e+18,
                        8.75e+18, 8.52e+18, 8.33e+18, 8.15e+18, 8.00e+18])
    Te_init = np.array([1.50, 1.33, 1.17, 1.04, 0.92, 0.82, 0.72, 0.64,
                        0.57, 0.50, 0.45, 0.40, 0.36, 0.32, 0.28, 0.25,
                        0.23, 0.20, 0.18, 0.16, 0.15, 0.13, 0.12, 0.11, 0.10])

    if args.init_dataset_only:
        preflight_required_inputs(cwd, eqdsk_list)
        print(f'Seed EQDSKs: {os.path.dirname(eqdsk_list[0])}')
        print(f'Dataset initialized: {os.path.abspath(args.output_dir)}')
        sys.exit(0)

    validate_jax_backend(require_cuda_on_gpu=require_cuda_on_gpu)

    initial_relax_cache = None
    if not args.no_initial_relax_cache:
        initial_relax_cache = args.initial_relax_cache
        if initial_relax_cache is None:
            initial_relax_cache = os.path.join(args.output_dir, 'initial_relax_state.json')
        initial_relax_cache = os.path.abspath(initial_relax_cache)

    needs_relax_cache_build = (
        initial_relax_cache is not None
        and (args.build_initial_relax_cache_only or not os.path.exists(initial_relax_cache))
    )
    preflight_required_inputs(
        cwd,
        eqdsk_list,
        initial_relax_cache=initial_relax_cache,
        require_initial_relax_cache=initial_relax_cache is not None and not needs_relax_cache_build,
    )
    print(f'Seed EQDSKs: {os.path.dirname(eqdsk_list[0])}')

    needs_main_tokamaker = args.n_workers == 1 or needs_relax_cache_build
    if needs_main_tokamaker:
        # ── One-time setup ────────────────────────────────────────────────────
        print('Setting up TokaMaker...')
        mygs, R0, B0, Z0 = setup_tokamaker(cwd)
    else:
        print(f'Setting up TokaMaker inside {args.n_workers} worker processes...')
        mygs = None

    end_idx = args.n_trajectories if args.end_idx is None else args.end_idx
    if not (0 <= args.start_idx <= end_idx <= args.n_trajectories):
        raise ValueError(
            f'Invalid range: require 0 <= start_idx <= end_idx <= n_trajectories, '
            f'got start_idx={args.start_idx}, end_idx={end_idx}, '
            f'n_trajectories={args.n_trajectories}'
        )

    run_ids = list(range(args.start_idx, end_idx))

    record_task_status(
        chunk_dir,
        {
            'status': 'running',
            'start_idx': args.start_idx,
            'end_idx': end_idx,
            'n_workers': args.n_workers,
            'output_dir': os.path.abspath(args.output_dir),
            'trajectories_dir': str(dataset_paths(args.output_dir)['trajectories']),
            'full_trajectories_dir': str(dataset_paths(args.output_dir)['full_trajectories']),
            'save_full_zarr': bool(args.save_full_zarr),
            'save_json': bool(args.save_json),
        },
    )

    if initial_relax_cache is not None and (run_ids or args.build_initial_relax_cache_only):
        cache_action_idx = args.start_idx if args.start_idx < args.n_trajectories else 0
        build_initial_relax_cache(
            mygs, all_actions[cache_action_idx], initial_relax_cache,
            eqdsk_list, eqtimes, coil_bounds, x_points,
            Ip_targets, ne_init, Te_init, psi_sample,
            log_dir=log_dir,
            grid_size=args.grid_size,
        )

    if args.build_initial_relax_cache_only:
        print(f'Initial relax cache ready: {initial_relax_cache}')
        sys.exit(0)

    mode = 'serially' if args.n_workers == 1 else f'across {args.n_workers} workers'
    print(f'Launching {len(run_ids)} trajectories {mode}...\n')

    t_start_total = time.time()
    success_count, fail_count = 0, 0

    if args.n_workers == 1:
        # ── Serial run loop ───────────────────────────────────────────────────
        for run_id in run_ids:
            if args.save_json:
                existing_path = trajectory_path(args.output_dir, run_id)
                if existing_path.exists():
                    raise FileExistsError(
                        f'trajectory output already exists before run starts: {existing_path}'
                    )
            if args.save_full_zarr:
                existing_zarr_path = full_trajectory_zarr_path(args.output_dir, run_id)
                if existing_zarr_path.exists():
                    raise FileExistsError(
                        f'full trajectory output already exists before run starts: {existing_zarr_path}'
                    )

            action_row = all_actions[run_id]

            print(f'\n[{run_id + 1}/{args.n_trajectories}] Running trajectory {run_id}...')
            t0 = time.time()

            transitions, summary, data_tree = run_single_trajectory(
                mygs, action_row, run_id, cwd, eqdsk_list, eqtimes,
                coil_bounds, x_points, diverted_isoflux_pts,
                Ip_targets, ne_init, Te_init, psi_sample,
                initial_relax_cache=initial_relax_cache,
                log_dir=log_dir,
                max_loop=args.max_loop,
                grid_size=args.grid_size,
                trajectory_timeout_seconds=args.trajectory_timeout_seconds,
            )

            elapsed = time.time() - t0

            if transitions is not None:
                path = save_trajectory_outputs(
                    transitions, summary, action_row, run_id, args.output_dir,
                    data_tree=data_tree,
                    save_full_zarr=args.save_full_zarr,
                    save_json=args.save_json,
                )
                success_count += 1
                print(f'  Saved to {path} ({elapsed:.1f}s)')
            else:
                fail_count += 1
                fail_path = save_failure_atomic(
                    args.output_dir,
                    run_id,
                    'simulation returned None',
                    chunk_dir=chunk_dir,
                )
                print(f'  Failed run logged to {fail_path} ({elapsed:.1f}s)')

            elapsed_total = time.time() - t_start_total
            done = success_count + fail_count
            eta = (elapsed_total / done) * (len(run_ids) - done) / 60 if done else 0.0
            print(f'  Progress: {success_count} ok, {fail_count} failed | '
                  f'Elapsed: {elapsed_total/60:.1f} min | ETA: {eta:.1f} min')
    else:
        worker = partial(
            worker_fn,
            all_actions=all_actions,
            eqdsk_list=eqdsk_list,
            eqtimes=eqtimes,
            coil_bounds=coil_bounds,
            x_points=x_points,
            diverted_isoflux_pts=diverted_isoflux_pts,
            Ip_targets=Ip_targets,
            ne_init=ne_init,
            Te_init=Te_init,
            psi_sample=psi_sample,
            dataset_dir=args.output_dir,
            initial_relax_cache=initial_relax_cache,
            log_dir=log_dir,
            max_loop=args.max_loop,
            grid_size=args.grid_size,
            trajectory_timeout_seconds=args.trajectory_timeout_seconds,
            save_full_zarr=args.save_full_zarr,
            save_json=args.save_json,
        )

        mp_context = os.environ.get('MP_CONTEXT', 'fork')
        ctx = mp.get_context(mp_context)
        with ctx.Pool(
            processes=args.n_workers,
            initializer=worker_init,
            initargs=(cwd, require_cuda_on_gpu),
        ) as pool:
            for run_id, ok, result in pool.imap_unordered(worker, run_ids):
                elapsed_total = time.time() - t_start_total
                done = success_count + fail_count + 1

                if ok:
                    success_count += 1
                    print(f'  [{run_id}] OK -> {result}')
                else:
                    fail_count += 1
                    fail_path = save_failure_atomic(
                        args.output_dir,
                        run_id,
                        result,
                        chunk_dir=chunk_dir,
                    )
                    print(f'  [{run_id}] FAILED: {result}; logged to {fail_path}')

                eta = (elapsed_total / done) * (len(run_ids) - done) / 60 if done else 0.0
                print(f'  Progress: {success_count} ok, {fail_count} failed | '
                      f'Elapsed: {elapsed_total/60:.1f} min | ETA: {eta:.1f} min')

    total = time.time() - t_start_total
    expected_count = len(run_ids)
    print(f'\nDone. {success_count}/{expected_count} requested trajectories saved '
          f'to {args.output_dir} in {total/60:.1f} min')

    record_task_status(
        chunk_dir,
        {
            'status': 'complete' if fail_count == 0 else 'failed',
            'success_count': success_count,
            'fail_count': fail_count,
            'start_idx': args.start_idx,
            'end_idx': end_idx,
            'n_workers': args.n_workers,
            'output_dir': os.path.abspath(args.output_dir),
            'trajectories_dir': str(dataset_paths(args.output_dir)['trajectories']),
            'full_trajectories_dir': str(dataset_paths(args.output_dir)['full_trajectories']),
            'save_full_zarr': bool(args.save_full_zarr),
            'save_json': bool(args.save_json),
        },
    )

    sys.stdout.flush()
    sys.stderr.flush()
    # Some JAX/TORAX cleanup paths can alter the process status after useful
    # work has completed. Return the dataset status explicitly for Slurm.
    os._exit(0 if fail_count == 0 else 1)
