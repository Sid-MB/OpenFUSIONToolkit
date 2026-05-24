"""
collect_trajectories.py
=======================
Generates a static offline RL dataset by running TokaMaker_TORAX simulations
with varied ECRH and NBI heating schedules sampled via Latin Hypercube Sampling.

Dataset structure (per trajectory, 12 transitions):
    s       : state vector at rl_time[i]
    a       : [ecrh_MW, nbi_MW] applied from rl_time[i] to rl_time[i+1]
    r       : mean Q_fusion from rl_time[i] to rl_time[i+1]
              (plus terminal reward at last step)
    s_next  : state vector at rl_time[i+1]
    done    : True at last transition

Usage:
    python collect_trajectories.py --n_trajectories 500 --output_dir ./rl_dataset
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.stats import qmc
from datetime import datetime
import io
from contextlib import redirect_stdout


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
    action_row[:, 1] = nbi_MW at each decision time.
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
    except Exception:
        return float('nan')


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

    # Check which rho coordinate THIS SPECIFIC VARIABLE uses
    var_dims = ds[var_name].dims
    print(f"DEBUG: var={var_name}, dims={var_dims}")  # DIAGNOSTIC

    if 'rho_face_norm' in var_dims:
        rho_coord = ds.coords['rho_face_norm'].values
        print(f"  -> using rho_face_norm, shape={rho_coord.shape}")
    elif 'rho_norm' in var_dims:
        rho_coord = ds.coords['rho_norm'].values
        print(f"  -> using rho_norm, shape={rho_coord.shape}")
    elif 'rho_cell_norm' in var_dims:
        rho_coord = ds.coords['rho_cell_norm'].values
        print(f"  -> using rho_cell_norm, shape={rho_coord.shape}")
    else:
        print(f"  -> ERROR: no rho coordinate found!")
        return [float('nan')] * len(rho_values)

    profile_at_t = profile_data[t_idx, :]
    print(f"  profile_at_t shape={profile_at_t.shape}, first 5 vals={profile_at_t[:5]}")

    result = [float(np.interp(rho, rho_coord, profile_at_t)) for rho in rho_values]
    print(f"  result for rho={rho_values}: {result}")
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
    except Exception:
        state['penalty_q95']   = 0.0
        state['penalty_betaN'] = 0.0
        state['penalty_fgw']   = 0.0

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
        except Exception:
            pass

    reward = step_reward - penalty

    # Terminal bonus
    if is_terminal:
        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()

        Q_avg   = summary.get('Q_flattop_avg', 0.0)
        flux_wb = summary.get('flux_consumed_Wb', 0.0)
        reward += Q_avg - FLUX_WEIGHT * flux_wb

    return reward


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
        a  = action_row[i].tolist()  # [ecrh_MW, nbi_MW]
        s = extract_state(tmtx, t, t_next, a)
        r = compute_reward(tmtx, t, t_next, is_terminal=is_terminal)

        transitions.append({
            's':      s,
            'a':      a,          # [ecrh_MW, nbi_MW] in MW
            'r':      r,
            'done':   is_terminal,
            't':      t,
            't_next': t_next,
        })

    return transitions


# ── Main simulation setup ─────────────────────────────────────────────────────

def setup_tokamaker(cwd):
    """Initialize OFT and TokaMaker, produce seed eqdsks. Run once per process."""

    import sys
    sys.path.append('/Users/deniz/Desktop/Spring2026/CS224R/project/OpenFUSIONToolkit/install_release/python')

    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux
    import numpy as np

    R0, B0, Z0 = 6.3, 5.2, 0.5

    myOFT = OFT_env(nthreads=1)
    mygs  = TokaMaker(myOFT)

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh('ITER_mesh.h5')
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.maxits = 100
    mygs.setup(order=2, F0=R0 * B0)
    mygs.set_coil_vsc({'VS': 1.0})

    return mygs, R0, B0, Z0


def run_single_trajectory(mygs, action_row, run_id, cwd, eqdsk_list, eqtimes,
                           coil_bounds, x_points, diverted_isoflux_pts,
                           Ip_targets, ne_init, Te_init, psi_sample):
    """
    Configure and run one TokaMaker_TORAX simulation with the given action_row.
    Returns the transitions list and summary dict, or None if simulation failed.
    """
    from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX
    import numpy as np

    ecrh_schedule = build_ecrh_schedule(action_row)
    nbi_schedule  = build_nbi_schedule(action_row)

    try:
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

        tmtx.set_TORAX_grid(grid_type='n_rho', grid=51)

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

        tmtx.fly(
            output_mode=False,
            max_loop=2,
            run_name='tmp',
            t_ave_toggle='flattop',
            t_ave_window=25,
            relax=True,
            relax_duration=5,
        )

        transitions = build_trajectory(tmtx, action_row)
        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()

        return transitions, summary

    except Exception as e:
        print(f'  [run {run_id}] FAILED: {e}')
        return None, None


# ── Dataset saving ────────────────────────────────────────────────────────────

def save_trajectory(transitions, summary, action_row, run_id, output_dir):
    """Save one trajectory as a JSON file."""
    payload = {
        'run_id':      run_id,
        'timestamp':   datetime.now().isoformat(),
        'actions_raw': action_row.tolist(),   # (21, 2) array in MW
        'transitions': transitions,            # list of 12 dicts
        'summary':     summary,
    }
    path = os.path.join(output_dir, f'trajectory_{run_id:04d}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajectories', type=int, default=500)
    parser.add_argument('--output_dir',     type=str, default='./rl_dataset')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--start_idx',      type=int, default=0,
                        help='Resume from this trajectory index')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cwd = os.getcwd()

    print(f'Sampling {args.n_trajectories} trajectories with LHS (seed={args.seed})')
    all_actions = sample_actions_lhs(args.n_trajectories, seed=args.seed)
    np.save(os.path.join(args.output_dir, 'all_actions.npy'), all_actions)
    print(f'Action matrix saved: shape {all_actions.shape}')

    # ── One-time setup ────────────────────────────────────────────────────────
    print('Setting up TokaMaker...')
    mygs, R0, B0, Z0 = setup_tokamaker(cwd)

    # ── Fixed simulation parameters ───────────────────────────────────────────
    coil_bounds = {key: [-50.e6, 50.e6] for key in [
        'CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L',
        'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6', 'VS'
    ]}

    Ip_targets = [1.5e6, 5e6, 15e6, 15e6, 1.5e6]
    eqdsk_list = [os.path.join(cwd, f'i={i}.eqdsk') for i in range(5)]
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

    run_ids = list(range(args.start_idx, args.n_trajectories))
    print(f'Launching {len(run_ids)} trajectories serially...\n')

    t_start_total = time.time()
    success_count, fail_count = 0, 0

    # ── Serial run loop ───────────────────────────────────────────────────────
    for run_id in run_ids:
        action_row = all_actions[run_id]

        print(f'\n[{run_id + 1}/{args.n_trajectories}] Running trajectory {run_id}...')
        t0 = time.time()

        transitions, summary = run_single_trajectory(
            mygs, action_row, run_id, cwd, eqdsk_list, eqtimes,
            coil_bounds, x_points, diverted_isoflux_pts,
            Ip_targets, ne_init, Te_init, psi_sample,
        )

        elapsed = time.time() - t0

        if transitions is not None:
            path = save_trajectory(transitions, summary, action_row, run_id, args.output_dir)
            success_count += 1
            print(f'  Saved to {path} ({elapsed:.1f}s)')
        else:
            fail_count += 1
            fail_log = os.path.join(args.output_dir, 'failed_runs.txt')
            with open(fail_log, 'a') as f:
                f.write(f'{run_id}\n')
            print(f'  Failed run logged ({elapsed:.1f}s)')

        elapsed_total = time.time() - t_start_total
        done = success_count + fail_count
        eta = (elapsed_total / done) * (len(run_ids) - done) / 60 if done else 0.0
        print(f'  Progress: {success_count} ok, {fail_count} failed | '
              f'Elapsed: {elapsed_total/60:.1f} min | ETA: {eta:.1f} min')

    total = time.time() - t_start_total
    print(f'\nDone. {success_count}/{args.n_trajectories} saved to {args.output_dir} '
          f'in {total/60:.1f} min')
