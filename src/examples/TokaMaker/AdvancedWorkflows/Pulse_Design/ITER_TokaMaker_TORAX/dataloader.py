import copy
import ast
import json
import os
import shutil
import tempfile
import sys
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


DATASET_SCHEMA_VERSION = 1
MANIFEST_FILENAME = 'run_manifest.json'
ACTIONS_FILENAME = 'all_actions.npy'
TRAJECTORIES_DIRNAME = 'trajectories'
FULL_TRAJECTORIES_DIRNAME = 'full_trajectories'
REPLAY_SHARDS_DIRNAME = 'replay_shards'
FAILURES_DIRNAME = 'failures'
CHUNKS_DIRNAME = 'chunks'
REWARD_RECALC_STATS_DIRNAME = 'reward_recalc_stats'
REPLAY_CACHE_DIRNAME = 'replay_cache'
REPLAY_CACHE_VERSION = 1


class DatasetMismatchError(RuntimeError):
    """Raised when an existing dataset does not match the requested run."""


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def dataset_paths(dataset_dir):
    root = Path(dataset_dir).resolve()
    return {
        'root': root,
        'manifest': root / MANIFEST_FILENAME,
        'actions': root / ACTIONS_FILENAME,
        'trajectories': root / TRAJECTORIES_DIRNAME,
        'full_trajectories': root / FULL_TRAJECTORIES_DIRNAME,
        'replay_shards': root / REPLAY_SHARDS_DIRNAME,
        'failures': root / FAILURES_DIRNAME,
        'chunks': root / CHUNKS_DIRNAME,
        'reward_recalc_stats': root / REWARD_RECALC_STATS_DIRNAME,
        'replay_cache': root / REPLAY_CACHE_DIRNAME,
    }


def ensure_dataset_dirs(dataset_dir):
    paths = dataset_paths(dataset_dir)
    paths['root'].mkdir(parents=True, exist_ok=True)
    paths['trajectories'].mkdir(parents=True, exist_ok=True)
    paths['full_trajectories'].mkdir(parents=True, exist_ok=True)
    paths['replay_shards'].mkdir(parents=True, exist_ok=True)
    paths['failures'].mkdir(parents=True, exist_ok=True)
    paths['chunks'].mkdir(parents=True, exist_ok=True)
    paths['reward_recalc_stats'].mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Per-run reward-weight overrides. Setting any of these env vars overrides that
# single field of the canonical RLRewardConfig for this process only, so a reward
# recalc / training / eval run can use different weights WITHOUT editing the
# checked-in defaults. Unset vars leave the canonical value untouched. Example:
#   RL_Q95_PENALTY_WEIGHT=3.5 RL_FGW_PENALTY_WEIGHT=4.0
_REWARD_ENV_OVERRIDES = {
    'q95_min': 'RL_Q95_MIN',
    'beta_n_max': 'RL_BETA_N_MAX',
    'fgw_max': 'RL_FGW_MAX',
    'step_reward_weight': 'RL_STEP_REWARD_WEIGHT',
    'q95_penalty_weight': 'RL_Q95_PENALTY_WEIGHT',
    'beta_n_penalty_weight': 'RL_BETA_N_PENALTY_WEIGHT',
    'fgw_penalty_weight': 'RL_FGW_PENALTY_WEIGHT',
    'q_flattop_weight': 'RL_Q_FLATTOP_WEIGHT',
    'flux_weight': 'RL_FLUX_WEIGHT',
}


def _apply_reward_env_overrides(values):
    for field, env_name in _REWARD_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != '':
            values[field] = float(raw)
    return values


def _base_reward_config():
    source_path = Path(__file__).resolve().parents[6] / 'src/python/OpenFUSIONToolkit/TokaMaker/pulse_design.py'
    if source_path.is_file():
        try:
            tree = ast.parse(source_path.read_text())
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'RLRewardConfig':
                    values = {}
                    for stmt in node.body:
                        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                            target = stmt.target.id
                            if isinstance(stmt.value, ast.Constant):
                                values[target] = stmt.value.value
                    if values:
                        return values
        except Exception as exc:
            print(
                f"WARNING: Could not parse reward defaults from {source_path}: {exc}. "
                "Falling back to imported RLRewardConfig or literal defaults.",
                file=sys.stderr,
            )
    try:
        from OpenFUSIONToolkit.TokaMaker.pulse_design import RLRewardConfig
        return asdict(RLRewardConfig())
    except Exception:
        return {
            'q95_min': 3.0,
            'beta_n_max': 2.8,
            'fgw_max': 0.85,
            'step_reward_weight': 1.0,
            'q95_penalty_weight': 1.2,
            'beta_n_penalty_weight': 1.0,
            'fgw_penalty_weight': 2.0,
            'q_flattop_weight': 1.0,
            'flux_weight': 0.012,
        }


def default_reward_config():
    return _apply_reward_env_overrides(_base_reward_config())


def reward_config_to_dict(reward_config=None):
    """Return a JSON-serializable reward config dict.

    Accepts the existing dict form, dataclass instances such as
    ``RLRewardConfig()``, or ``None`` for the current defaults.
    """
    if reward_config is None:
        return default_reward_config()
    if isinstance(reward_config, dict):
        return copy.deepcopy(reward_config)
    if hasattr(reward_config, '__dataclass_fields__'):
        return copy.deepcopy(asdict(reward_config))
    if hasattr(reward_config, '__dict__'):
        return copy.deepcopy({
            key: value
            for key, value in vars(reward_config).items()
            if not key.startswith('_')
        })
    return copy.deepcopy(reward_config)


def stable_json_dumps(payload):
    return json.dumps(
        _normalize_for_json(payload),
        indent=2,
        sort_keys=True,
        separators=(',', ': '),
    )


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix=f'.tmp.{os.getpid()}',
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(stable_json_dumps(payload))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix=f'.tmp.{os.getpid()}',
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(stable_json_dumps(payload))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise FileExistsError(f'File already exists: {path}') from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_save_npy_once(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix=f'.tmp.{os.getpid()}',
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise FileExistsError(f'File already exists: {path}') from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_save_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix=f'.tmp.{os.getpid()}',
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise FileExistsError(f'File already exists: {path}') from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def create_run_manifest(*, n_trajectories, seed, max_loop, grid_size,
                        decision_times, rl_times, action_bounds,
                        sampler, observation_mode='legacy',
                        reward_config=None,
                        start_idx=None, end_idx=None):
    return {
        'schema_version': DATASET_SCHEMA_VERSION,
        'created_at': utc_now_iso(),
        'n_trajectories': int(n_trajectories),
        'seed': int(seed),
        'max_loop': int(max_loop),
        'grid_size': int(grid_size),
        'observation_mode': str(observation_mode),
        'decision_times': [int(t) for t in decision_times],
        'rl_times': [int(t) for t in rl_times],
        'action_bounds': copy.deepcopy(action_bounds),
        'sampler': copy.deepcopy(sampler),
        'reward_config': copy.deepcopy(reward_config or default_reward_config()),
        'requested_range': {
            'start_idx': None if start_idx is None else int(start_idx),
            'end_idx': None if end_idx is None else int(end_idx),
        },
        'layout': {
            'actions': ACTIONS_FILENAME,
            'trajectories': TRAJECTORIES_DIRNAME,
            'full_trajectories': FULL_TRAJECTORIES_DIRNAME,
            'replay_shards': REPLAY_SHARDS_DIRNAME,
            'failures': FAILURES_DIRNAME,
            'chunks': CHUNKS_DIRNAME,
            'reward_recalc_stats': REWARD_RECALC_STATS_DIRNAME,
        },
    }


def manifest_comparison_subset(manifest):
    # Older manifests predate observation_mode; treat them as legacy so
    # existing datasets remain reusable when the default schema is requested.
    observation_mode = manifest.get('observation_mode', 'legacy')
    keys = (
        'schema_version',
        'n_trajectories',
        'seed',
        'max_loop',
        'grid_size',
        'observation_mode',
        'reward_config',
        'decision_times',
        'rl_times',
        'action_bounds',
        'sampler',
    )
    subset = {key: manifest.get(key) for key in keys}
    subset['observation_mode'] = observation_mode
    subset['reward_config'] = copy.deepcopy(manifest.get('reward_config', default_reward_config()))
    return subset


def compare_manifests(existing, expected):
    existing_subset = manifest_comparison_subset(existing)
    expected_subset = manifest_comparison_subset(expected)
    mismatches = []
    for key in expected_subset:
        if existing_subset.get(key) != expected_subset.get(key):
            mismatches.append(
                f'{key}: existing={existing_subset.get(key)!r}, '
                f'expected={expected_subset.get(key)!r}'
            )
    return mismatches


def initialize_dataset(dataset_dir, expected_manifest, actions):
    paths = ensure_dataset_dirs(dataset_dir)

    if paths['manifest'].exists():
        manifest = load_json(paths['manifest'])
        validate_manifest(manifest, expected_manifest, paths['manifest'])
    else:
        try:
            atomic_write_json_once(paths['manifest'], expected_manifest)
            manifest = expected_manifest
        except FileExistsError:
            manifest = load_json(paths['manifest'])
            validate_manifest(manifest, expected_manifest, paths['manifest'])

    if paths['actions'].exists():
        existing_actions = np.load(paths['actions'])
        validate_actions(existing_actions, actions, paths['actions'])
    else:
        try:
            atomic_save_npy_once(paths['actions'], actions)
        except FileExistsError:
            existing_actions = np.load(paths['actions'])
            validate_actions(existing_actions, actions, paths['actions'])

    return manifest


def require_dataset(dataset_dir, expected_manifest, expected_actions=None):
    paths = dataset_paths(dataset_dir)
    if not paths['manifest'].is_file():
        raise FileNotFoundError(
            f'Dataset manifest is missing: {paths["manifest"]}. '
            'Initialize the dataset before submitting array workers.'
        )
    if not paths['actions'].is_file():
        raise FileNotFoundError(
            f'Action table is missing: {paths["actions"]}. '
            'Initialize the dataset before submitting array workers.'
        )

    manifest = load_json(paths['manifest'])
    validate_manifest(manifest, expected_manifest, paths['manifest'])
    actions = np.load(paths['actions'])
    if expected_actions is not None:
        validate_actions(actions, expected_actions, paths['actions'])
    return manifest, actions


def validate_manifest(existing, expected, path):
    mismatches = compare_manifests(existing, expected)
    if mismatches:
        details = '\n  '.join(mismatches)
        raise DatasetMismatchError(f'Dataset manifest mismatch at {path}:\n  {details}')


def validate_actions(existing, expected, path):
    if existing.shape != expected.shape:
        raise DatasetMismatchError(
            f'Action table shape mismatch at {path}: '
            f'existing={existing.shape}, expected={expected.shape}'
        )
    if not np.allclose(existing, expected, rtol=0.0, atol=0.0):
        max_abs = float(np.max(np.abs(existing - expected)))
        raise DatasetMismatchError(
            f'Action table mismatch at {path}: max_abs_diff={max_abs}'
        )
    if not np.all(np.isfinite(existing)):
        raise DatasetMismatchError(f'Action table contains non-finite values: {path}')


def trajectory_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['trajectories'] / f'trajectory_{int(run_id):04d}.json'


def full_trajectory_zarr_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['full_trajectories'] / f'trajectory_{int(run_id):04d}.zarr'


def reward_recalc_stats_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['reward_recalc_stats'] / f'trajectory_{int(run_id):04d}.npz'


def replay_shard_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['replay_shards'] / f'trajectory_{int(run_id):04d}.npz'


def failure_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['failures'] / f'failed_run_{int(run_id):04d}.json'


def save_trajectory_atomic(dataset_dir, payload):
    run_id = int(payload['run_id'])
    path = trajectory_path(dataset_dir, run_id)
    atomic_write_json_once(path, payload)
    return str(path)


def save_reward_recalc_stats_atomic(dataset_dir, payload, data_tree):
    run_id = int(payload['run_id'])
    path = reward_recalc_stats_path(dataset_dir, run_id)
    if path.exists():
        raise FileExistsError(f'File already exists: {path}')
    if data_tree is None:
        raise ValueError('save_stats_for_reward_recalc=True requires a TORAX data_tree')

    scalars = data_tree['scalars']
    summary = payload.get('summary') or {}
    reward_total = float(payload.get('reward_total', np.sum([float(t.get('r', 0.0)) for t in payload.get('transitions', [])])))
    arrays = {
        'torax_time': np.asarray(scalars.coords['time'].values, dtype=np.float32),
        'decision_times': np.asarray(payload.get('decision_times', []), dtype=np.int32),
        'rl_times': np.asarray(payload.get('rl_times', []), dtype=np.int32),
        'Q_fusion': np.asarray(scalars['Q_fusion'].values, dtype=np.float32),
        'q95': np.asarray(scalars['q95'].values, dtype=np.float32),
        'beta_N': np.asarray(scalars['beta_N'].values, dtype=np.float32),
        'fgw_n_e_line_avg': np.asarray(scalars['fgw_n_e_line_avg'].values, dtype=np.float32),
        'terminal_Q_flattop_avg': np.asarray([float(summary.get('Q_flattop_avg', 0.0))], dtype=np.float32),
        'terminal_flux_consumed_Wb': np.asarray([float(summary.get('flux_consumed_Wb', 0.0))], dtype=np.float32),
        'reward_total': np.asarray([reward_total], dtype=np.float32),
        'run_id': np.asarray(run_id, dtype=np.int64),
        'timestamp': np.asarray(str(payload.get('timestamp') or ''), dtype=np.str_),
        'reward_config_json': np.asarray(
            json.dumps(reward_config_to_dict(payload.get('reward_config')), sort_keys=True),
            dtype=np.str_,
        ),
    }
    atomic_save_npz_once(path, **arrays)
    return str(path)


def load_reward_recalc_stats(stats_path):
    with np.load(stats_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def recompute_reward_series_from_stats(stats, rl_times, reward_config=None, actions_per_decision=None):
    """Recompute the per-decision reward series from saved scalar traces.

    The saved stats bundle is intentionally compact: it stores the TORAX scalar
    traces needed to re-score each RL interval without rerunning the physics.

    actions_per_decision: optional array shape (n_decisions, 2) of [ecrh_W, nbi_W]
      per decision step. When provided and RL_REWARD_MODE=pfusion, the step reward
      uses log(mean_Q * P_aux_MW + 1) (P_fusion proxy) instead of log(mean_Q + 1),
      removing the 1/P_aux blow-up exploit.
    """
    if isinstance(stats, (str, Path)):
        stats = load_reward_recalc_stats(stats)

    cfg = reward_config_to_dict(reward_config)
    boundaries = np.asarray(rl_times, dtype=float).ravel()
    if boundaries.size < 2:
        raise ValueError('rl_times must contain at least two interval boundaries')

    torax_time = np.asarray(stats['torax_time'], dtype=float).ravel()
    q_fusion = np.asarray(stats['Q_fusion'], dtype=float).ravel()
    q95 = np.asarray(stats['q95'], dtype=float).ravel()
    beta_n = np.asarray(stats['beta_N'], dtype=float).ravel()
    fgw = np.asarray(stats['fgw_n_e_line_avg'], dtype=float).ravel()
    q_flattop_avg = float(np.asarray(stats['terminal_Q_flattop_avg'], dtype=float).reshape(-1)[0])
    flux_consumed_wb = float(np.asarray(stats['terminal_flux_consumed_Wb'], dtype=float).reshape(-1)[0])

    # Anti-reward-hacking: clamp Q so starving aux power (Q=Pfus/Paux -> inf) can't yield
    # unbounded reward. Env-gated, default OFF (RL_Q_CLAMP unset => no clamp). Must match
    # the clamp used by pulse_design.compute_rewards at eval time.
    try:
        q_clamp = float(os.environ.get('RL_Q_CLAMP', '') or 0.0)
    except ValueError:
        q_clamp = 0.0
    if q_clamp > 0:
        q_fusion = np.minimum(q_fusion, q_clamp)
        q_flattop_avg = min(q_flattop_avg, q_clamp)

    # RL_REWARD_MODE=pfusion: use log(Q*P_aux_MW + 1) proxy for P_fusion.
    # P_aux is the sum of ECRH+NBI actions for that decision (provided via actions_per_decision).
    # This removes the 1/P_aux blow-up: zeroing aux gives log(0+1)=0 rather than infinity.
    try:
        reward_mode = os.environ.get('RL_REWARD_MODE', '').lower().strip()
    except Exception:
        reward_mode = ''
    use_pfusion = (reward_mode == 'pfusion') and (actions_per_decision is not None)

    rewards = np.empty(boundaries.size - 1, dtype=np.float32)
    safety_start_time = float(boundaries[0])

    for idx, (t_start, t_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (torax_time >= t_start) & (torax_time <= t_end)
        if np.any(mask):
            q_mean = float(np.nanmean(q_fusion[mask]))
            if use_pfusion and idx < len(actions_per_decision):
                p_aux_mw = float(actions_per_decision[idx, 0] + actions_per_decision[idx, 1]) / 1e6
                step_reward = np.log(q_mean * p_aux_mw + 1.0)
            else:
                step_reward = np.log(q_mean + 1.0)
        else:
            step_reward = 0.0

        penalty = 0.0
        if t_start >= safety_start_time:
            penalty += float(cfg['q95_penalty_weight']) * float(np.sum(np.maximum(float(cfg['q95_min']) - q95[mask], 0.0)))
            penalty += float(cfg['beta_n_penalty_weight']) * float(np.sum(np.maximum(beta_n[mask] - float(cfg['beta_n_max']), 0.0)))
            penalty += float(cfg['fgw_penalty_weight']) * float(np.sum(np.maximum(fgw[mask] - float(cfg['fgw_max']), 0.0)))

        reward = float(cfg['step_reward_weight']) * step_reward - penalty
        if idx == boundaries.size - 2:
            reward += float(cfg['q_flattop_weight']) * q_flattop_avg - float(cfg['flux_weight']) * flux_consumed_wb
        rewards[idx] = np.float32(reward)

    return rewards


def _xarray_node_to_dataset(node):
    if isinstance(node, xr.Dataset):
        return node
    if hasattr(node, 'to_dataset'):
        return node.to_dataset()
    if hasattr(node, 'ds') and isinstance(node.ds, xr.Dataset):
        return node.ds
    raise TypeError(f'Cannot convert {type(node)!r} to xarray.Dataset')


def _write_xarray_group(dataset, store_path, group_name):
    dataset = dataset.copy()
    dataset.attrs = _normalize_for_json(dataset.attrs)
    for var_name in dataset.variables:
        dataset[var_name].attrs = _normalize_for_json(dataset[var_name].attrs)
    dataset.to_zarr(store_path, group=group_name, mode='a', consolidated=False)


def build_reward_components_dataset(transitions):
    decision_t = np.array([transition.get('t', np.nan) for transition in transitions], dtype=float)
    t_next = np.array([transition.get('t_next', np.nan) for transition in transitions], dtype=float)
    rewards = np.array([transition['r'] for transition in transitions], dtype=float)
    actions = np.array([transition['a'] for transition in transitions], dtype=float)
    dones = np.array([bool(transition.get('done', False)) for transition in transitions], dtype=bool)

    data_vars = {
        'reward': ('decision', rewards),
        'done': ('decision', dones),
        'action': (('decision', 'action_dim'), actions),
    }

    state_keys = []
    if transitions and isinstance(transitions[0].get('s'), dict):
        state_keys = list(transitions[0]['s'].keys())
        for key in state_keys:
            values = [transition.get('s', {}).get(key, np.nan) for transition in transitions]
            if all(isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)) for value in values):
                data_vars[f'state_{key}'] = ('decision', np.array(values, dtype=float))

            next_values = [transition.get('s_next', {}).get(key, np.nan) for transition in transitions]
            if all(isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)) for value in next_values):
                data_vars[f'next_state_{key}'] = ('decision', np.array(next_values, dtype=float))

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            'decision': np.arange(len(transitions), dtype=np.int32),
            'decision_t': ('decision', decision_t),
            't_next': ('decision', t_next),
            'action_dim': np.arange(actions.shape[1], dtype=np.int32),
        },
        attrs={
            'state_keys': state_keys,
            'has_explicit_next_state': all(
                f'next_state_{key}' in data_vars for key in state_keys
            ),
        },
    )
    return dataset


def transition_block_from_transitions(transitions, state_keys=None):
    if not transitions:
        keys = [] if state_keys is None else list(state_keys)
        return {
            'state_keys': keys,
            'has_explicit_next_state': True,
            'states': np.empty((0, len(keys)), dtype=np.float32),
            'actions': np.empty((0, 0), dtype=np.float32),
            'next_states': np.empty((0, len(keys)), dtype=np.float32),
            'rewards': np.empty((0, 1), dtype=np.float32),
            'dones': np.empty((0, 1), dtype=np.float32),
        }

    keys = list(state_keys) if state_keys is not None else list(transitions[0]['s'].keys())
    states = np.stack([load_state(transition['s'], keys) for transition in transitions])
    actions = np.asarray([transition['a'] for transition in transitions], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    rewards = np.asarray([transition['r'] for transition in transitions], dtype=np.float32).reshape(-1, 1)
    dones = np.asarray(
        [
            transition.get('done', idx == len(transitions) - 1)
            for idx, transition in enumerate(transitions)
        ],
        dtype=np.float32,
    ).reshape(-1, 1)

    has_explicit_next_state = all('s_next' in transition for transition in transitions)
    if has_explicit_next_state:
        next_states = np.stack([load_state(transition['s_next'], keys) for transition in transitions])
    else:
        next_states = np.zeros_like(states)
        if states.shape[0] > 1:
            not_terminal = dones[:-1, 0] < 0.5
            next_indices = np.nonzero(not_terminal)[0]
            next_states[next_indices] = states[next_indices + 1]

    return {
        'state_keys': keys,
        'has_explicit_next_state': has_explicit_next_state,
        'states': np.ascontiguousarray(states, dtype=np.float32),
        'actions': np.ascontiguousarray(actions, dtype=np.float32),
        'next_states': np.ascontiguousarray(next_states, dtype=np.float32),
        'rewards': np.ascontiguousarray(rewards, dtype=np.float32),
        'dones': np.ascontiguousarray(dones, dtype=np.float32),
    }


def save_replay_shard_atomic(dataset_dir, payload):
    run_id = int(payload['run_id'])
    path = replay_shard_path(dataset_dir, run_id)
    block = transition_block_from_transitions(payload['transitions'])
    atomic_save_npz_once(
        path,
        states=block['states'],
        actions=block['actions'],
        next_states=block['next_states'],
        rewards=block['rewards'],
        dones=block['dones'],
        state_keys=np.asarray(block['state_keys'], dtype=np.str_),
        has_explicit_next_state=np.asarray(block['has_explicit_next_state'], dtype=np.bool_),
        run_id=np.asarray(run_id, dtype=np.int64),
        actions_raw_json=np.asarray(json.dumps(payload.get('actions_raw')), dtype=np.str_),
        summary_json=np.asarray(json.dumps(payload.get('summary') or {}), dtype=np.str_),
        timestamp=np.asarray(str(payload.get('timestamp') or ''), dtype=np.str_),
    )
    return str(path)


def save_full_trajectory_zarr_atomic(dataset_dir, payload, data_tree, json_path=None):
    """Write one per-trajectory Zarr store and publish it with an atomic rename."""
    import zarr

    run_id = int(payload['run_id'])
    final_path = full_trajectory_zarr_path(dataset_dir, run_id)
    if final_path.exists():
        raise FileExistsError(f'File already exists: {final_path}')

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f'.{final_path.name}.tmp.{os.getpid()}')
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    try:
        for group_name in ('scalars', 'profiles'):
            if group_name in data_tree:
                _write_xarray_group(
                    _xarray_node_to_dataset(data_tree[group_name]),
                    tmp_path,
                    group_name,
                )

        reward_components = build_reward_components_dataset(payload['transitions'])
        _write_xarray_group(reward_components, tmp_path, 'reward_components')

        root = zarr.open_group(str(tmp_path), mode='a')
        attrs = {
            'schema_version': DATASET_SCHEMA_VERSION,
            'kind': 'tokamaker_torax_full_trajectory',
            'run_id': payload['run_id'],
            'timestamp': payload['timestamp'],
            'actions_raw': payload['actions_raw'],
            'summary': payload['summary'],
        }
        if json_path is not None:
            attrs['json_trajectory'] = str(json_path)
        root.attrs.update(_normalize_for_json(attrs))
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise

    return str(final_path)


def save_failure_atomic(dataset_dir, run_id, error, *, chunk_dir=None):
    payload = {
        'run_id': int(run_id),
        'timestamp': utc_now_iso(),
        'error': str(error),
        'chunk_dir': None if chunk_dir is None else str(Path(chunk_dir).resolve()),
    }
    path = failure_path(dataset_dir, run_id)
    atomic_write_json(path, payload)
    return str(path)


def record_task_status(chunk_dir, status):
    if chunk_dir is None:
        return None
    path = Path(chunk_dir).resolve() / 'task_status.json'
    payload = dict(status)
    payload['timestamp'] = utc_now_iso()
    atomic_write_json(path, payload)
    return str(path)


def find_trajectory_files(dataset_dir):
    root = Path(dataset_dir)
    nested = root / TRAJECTORIES_DIRNAME
    if nested.is_dir():
        files = sorted(nested.glob('trajectory_*.json'))
        if files:
            return files
    return sorted(root.glob('trajectory_*.json'))


def find_full_trajectory_zarr_stores(dataset_dir):
    root = Path(dataset_dir)
    nested = root / FULL_TRAJECTORIES_DIRNAME
    if nested.is_dir():
        stores = sorted(nested.glob('trajectory_*.zarr'))
        if stores:
            return stores
    return sorted(root.glob('trajectory_*.zarr'))


def find_replay_shard_files(dataset_dir):
    root = Path(dataset_dir)
    nested = root / REPLAY_SHARDS_DIRNAME
    if nested.is_dir():
        files = sorted(nested.glob('trajectory_*.npz'))
        if files:
            return files
    return sorted(root.glob('trajectory_*.npz'))


def load_state(state, state_keys=None):
    keys = state_keys if state_keys is not None else state.keys()
    return np.array([state[key] for key in keys], dtype=np.float32)


def _zarr_reward_components_dataset(store_path):
    return xr.open_zarr(store_path, group='reward_components', consolidated=False)


def _zarr_state_keys(dataset):
    state_keys = dataset.attrs.get('state_keys')
    if state_keys:
        return list(state_keys)
    return sorted(
        var_name.removeprefix('state_')
        for var_name in dataset.data_vars
        if var_name.startswith('state_') and not var_name.startswith('next_state_')
    )


def _zarr_has_explicit_next_state(dataset, state_keys=None):
    keys = list(state_keys) if state_keys is not None else _zarr_state_keys(dataset)
    return bool(keys) and all(f'next_state_{key}' in dataset for key in keys)


def _inspect_zarr_store_specs(store_path):
    dataset = _zarr_reward_components_dataset(store_path)
    try:
        n_decisions = int(dataset.sizes.get('decision', 0))
        if n_decisions == 0:
            return None

        keys = _zarr_state_keys(dataset)
        if not keys:
            return None

        return {
            'num_transitions': n_decisions,
            'state_keys': keys,
            'action_dim': int(dataset.sizes['action_dim']),
            'has_explicit_next_state': _zarr_has_explicit_next_state(dataset, keys),
        }
    finally:
        dataset.close()


def _replay_cache_executor(worker_backend):
    if worker_backend == 'process':
        return ProcessPoolExecutor
    if worker_backend == 'thread':
        return ThreadPoolExecutor
    raise ValueError(
        f'Unsupported worker_backend={worker_backend!r}; expected "process" or "thread".'
    )


def infer_zarr_dataset_specs(directory, *, show_progress=False, max_workers=1,
                             worker_backend='process'):
    zarr_stores = find_full_trajectory_zarr_stores(directory)
    if not zarr_stores:
        raise FileNotFoundError(f'No trajectory_*.zarr stores found in {directory}')

    total_transitions = 0
    state_keys = None
    action_dim = None
    nonempty_count = 0
    explicit_next_state_count = 0

    if max_workers > 1:
        executor_cls = _replay_cache_executor(worker_backend)
        progress = None
        if show_progress:
            from tqdm import tqdm

            progress = tqdm(total=len(zarr_stores), desc='scan zarr stores', unit='store')
        try:
            with executor_cls(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_inspect_zarr_store_specs, store_path)
                    for store_path in zarr_stores
                ]
                for future in as_completed(futures):
                    info = future.result()
                    if progress is not None:
                        progress.update(1)
                    if info is None:
                        continue

                    keys = info['state_keys']
                    if state_keys is None:
                        state_keys = keys
                        action_dim = int(info['action_dim'])

                    if info['has_explicit_next_state']:
                        explicit_next_state_count += 1

                    total_transitions += int(info['num_transitions'])
                    nonempty_count += 1
        finally:
            if progress is not None:
                progress.close()
    else:
        stores_iter = zarr_stores
        if show_progress:
            from tqdm import tqdm

            stores_iter = tqdm(zarr_stores, desc='scan zarr stores', unit='store')

        for store_path in stores_iter:
            info = _inspect_zarr_store_specs(store_path)
            if info is None:
                continue

            keys = info['state_keys']
            if state_keys is None:
                state_keys = keys
                action_dim = int(info['action_dim'])

            if info['has_explicit_next_state']:
                explicit_next_state_count += 1

            total_transitions += int(info['num_transitions'])
            nonempty_count += 1

    if state_keys is None or action_dim is None:
        raise ValueError(f'No non-empty Zarr trajectories found in {directory}')

    return {
        'num_trajectories': nonempty_count,
        'num_transitions': total_transitions,
        'state_dim': len(state_keys),
        'action_dim': action_dim,
        'state_keys': state_keys,
        'format': 'zarr',
        'explicit_next_state_store_count': explicit_next_state_count,
        'has_explicit_next_state': explicit_next_state_count == nonempty_count,
    }


def infer_json_dataset_specs(directory):
    trajectory_files = find_trajectory_files(directory)
    if not trajectory_files:
        raise FileNotFoundError(f'No trajectory_*.json files found in {directory}')

    total_transitions = 0
    state_keys = None
    action_dim = None
    for filepath in trajectory_files:
        traj = load_json(filepath)
        transitions = traj.get('transitions', [])
        if not transitions:
            continue

        if state_keys is None:
            state_keys = list(transitions[0]['s'].keys())
            action_dim = len(transitions[0]['a'])

        total_transitions += len(transitions)

    if state_keys is None or action_dim is None:
        raise ValueError(f'No non-empty trajectories found in {directory}')

    return {
        'num_trajectories': len(trajectory_files),
        'num_transitions': total_transitions,
        'state_dim': len(state_keys),
        'action_dim': action_dim,
        'state_keys': state_keys,
        'format': 'json',
        'explicit_next_state_store_count': 0,
        'has_explicit_next_state': True,
    }


def _load_replay_shard_block(shard_path, state_keys=None):
    with np.load(shard_path, allow_pickle=False) as data:
        keys = [str(key) for key in data['state_keys'].tolist()]
        if state_keys is not None and list(state_keys) != keys:
            raise ValueError(
                f'Replay shard state keys do not match requested state keys in {shard_path}: '
                f'shard={keys!r}, requested={list(state_keys)!r}'
            )
        return {
            'state_keys': keys,
            'has_explicit_next_state': bool(np.asarray(data['has_explicit_next_state']).item()),
            'states': np.ascontiguousarray(data['states'], dtype=np.float32),
            'actions': np.ascontiguousarray(data['actions'], dtype=np.float32),
            'next_states': np.ascontiguousarray(data['next_states'], dtype=np.float32),
            'rewards': np.ascontiguousarray(data['rewards'], dtype=np.float32),
            'dones': np.ascontiguousarray(data['dones'], dtype=np.float32),
        }


def infer_replay_shard_specs(directory):
    shard_files = find_replay_shard_files(directory)
    if not shard_files:
        raise FileNotFoundError(f'No trajectory_*.npz replay shards found in {directory}')

    total_transitions = 0
    state_keys = None
    action_dim = None
    nonempty_count = 0
    explicit_next_state_count = 0
    for shard_path in shard_files:
        block = _load_replay_shard_block(shard_path)
        keys = list(block['state_keys'])
        if state_keys is None:
            state_keys = keys
            action_dim = int(block['actions'].shape[1])
        elif state_keys != keys:
            raise ValueError(
                'Replay shard state keys differ within dataset: '
                f'expected={state_keys!r}, got={keys!r} in {shard_path}'
            )
        elif action_dim != int(block['actions'].shape[1]):
            raise ValueError(
                'Replay shard action_dim differs within dataset: '
                f'expected={action_dim}, got={block["actions"].shape[1]} in {shard_path}'
            )

        block_size = int(block['states'].shape[0])
        if block_size == 0:
            continue
        total_transitions += block_size
        nonempty_count += 1
        if block.get('has_explicit_next_state', False):
            explicit_next_state_count += 1

    if state_keys is None or action_dim is None:
        raise ValueError(f'No non-empty replay shards found in {directory}')

    return {
        'num_trajectories': nonempty_count,
        'num_transitions': total_transitions,
        'state_dim': len(state_keys),
        'action_dim': action_dim,
        'state_keys': state_keys,
        'format': 'replay_shards',
        'explicit_next_state_store_count': explicit_next_state_count,
        'has_explicit_next_state': explicit_next_state_count == nonempty_count,
    }


def infer_dataset_specs(directory, *, show_progress=False, max_workers=1,
                        worker_backend='process'):
    replay_shards = find_replay_shard_files(directory)
    if replay_shards:
        return infer_replay_shard_specs(directory)
    zarr_stores = find_full_trajectory_zarr_stores(directory)
    if zarr_stores:
        return infer_zarr_dataset_specs(
            directory,
            show_progress=show_progress,
            max_workers=max_workers,
            worker_backend=worker_backend,
        )
    return infer_json_dataset_specs(directory)


def describe_dataset(directory, *, show_progress=False, max_workers=1,
                     worker_backend='process'):
    replay_shards = find_replay_shard_files(directory)
    zarr_stores = find_full_trajectory_zarr_stores(directory)
    json_files = find_trajectory_files(directory)

    if replay_shards:
        specs = infer_replay_shard_specs(directory)
        selected_format = 'replay_shards'
    elif zarr_stores:
        specs = infer_zarr_dataset_specs(
            directory,
            show_progress=show_progress,
            max_workers=max_workers,
            worker_backend=worker_backend,
        )
        selected_format = 'zarr'
    elif json_files:
        specs = infer_json_dataset_specs(directory)
        selected_format = 'json'
    else:
        raise FileNotFoundError(
            f'No trajectory data found in {directory}: expected '
            f'{REPLAY_SHARDS_DIRNAME}/trajectory_*.npz, '
            f'{FULL_TRAJECTORIES_DIRNAME}/trajectory_*.zarr or '
            f'{TRAJECTORIES_DIRNAME}/trajectory_*.json'
        )

    description = dict(specs)
    description.update({
        'selected_format': selected_format,
        'replay_shard_count': len(replay_shards),
        'zarr_store_count': len(zarr_stores),
        'json_file_count': len(json_files),
        'zarr_takes_precedence': bool(zarr_stores and json_files),
        'dataset_has_explicit_next_state': specs.get('has_explicit_next_state', False),
        'zarr_explicit_next_state_store_count': specs.get('explicit_next_state_store_count', 0),
    })
    return description


def iter_json_d4rl_transitions(directory, state_keys=None):
    for filepath in find_trajectory_files(directory):
        traj = load_json(filepath)
        transitions = traj['transitions']
        for idx, transition in enumerate(transitions):
            state = load_state(transition['s'], state_keys)
            action = np.array(transition['a'], dtype=np.float32)
            if 's_next' in transition:
                next_state = load_state(transition['s_next'], state_keys)
            elif idx < len(transitions) - 1:
                next_state = load_state(transitions[idx + 1]['s'], state_keys)
            else:
                next_state = np.zeros_like(state)
            reward = np.array([transition['r']], dtype=np.float32)
            done = np.array([transition.get('done', idx == len(transitions) - 1)], dtype=np.float32)
            yield state, action, next_state, reward, done


def iter_zarr_d4rl_transitions(directory, state_keys=None):
    for store_path in find_full_trajectory_zarr_stores(directory):
        block = _load_zarr_transition_block(store_path, state_keys=state_keys)
        for idx in range(block['states'].shape[0]):
            yield (
                block['states'][idx],
                block['actions'][idx],
                block['next_states'][idx],
                block['rewards'][idx],
                block['dones'][idx],
            )


def iter_replay_shard_d4rl_transitions(directory, state_keys=None):
    for shard_path in find_replay_shard_files(directory):
        block = _load_replay_shard_block(shard_path, state_keys=state_keys)
        for idx in range(block['states'].shape[0]):
            yield (
                block['states'][idx],
                block['actions'][idx],
                block['next_states'][idx],
                block['rewards'][idx],
                block['dones'][idx],
            )


def iter_d4rl_transitions(directory, state_keys=None):
    if find_replay_shard_files(directory):
        yield from iter_replay_shard_d4rl_transitions(directory, state_keys)
    elif find_full_trajectory_zarr_stores(directory):
        yield from iter_zarr_d4rl_transitions(directory, state_keys)
    else:
        yield from iter_json_d4rl_transitions(directory, state_keys)


def load_d4rl_dataset(directory, buffer, state_keys=None, cache_dir=None, prefer_cache=True):
    cache_dir = replay_cache_path(directory, cache_dir=cache_dir)
    if prefer_cache and replay_cache_exists(cache_dir):
        load_replay_cache_into_buffer(cache_dir, buffer, state_keys=state_keys)
        return

    for state, action, next_state, reward, done in iter_d4rl_transitions(directory, state_keys):
        buffer.add(state, action, next_state, reward, done)


def replay_cache_path(dataset_dir, cache_dir=None):
    if cache_dir is not None:
        return Path(cache_dir).resolve()
    return dataset_paths(dataset_dir)['replay_cache']


def replay_cache_files(cache_dir):
    cache_dir = Path(cache_dir)
    return {
        'manifest': cache_dir / 'replay_manifest.json',
        'states': cache_dir / 'states.npy',
        'actions': cache_dir / 'actions.npy',
        'next_states': cache_dir / 'next_states.npy',
        'rewards': cache_dir / 'rewards.npy',
        'dones': cache_dir / 'dones.npy',
    }


def replay_cache_exists(cache_dir):
    files = replay_cache_files(cache_dir)
    return all(path.is_file() for path in files.values())


def load_replay_cache_manifest(cache_dir):
    files = replay_cache_files(cache_dir)
    if not files['manifest'].is_file():
        raise FileNotFoundError(f'Replay cache manifest missing: {files["manifest"]}')
    return load_json(files['manifest'])


def infer_replay_cache_specs(cache_dir):
    manifest = load_replay_cache_manifest(cache_dir)
    state_keys = list(manifest['state_keys'])
    return {
        'num_trajectories': int(manifest.get('num_trajectories', 0)),
        'num_transitions': int(manifest['num_transitions']),
        'state_dim': int(manifest['state_dim']),
        'action_dim': int(manifest['action_dim']),
        'state_keys': state_keys,
        'format': 'replay_cache',
        'cache_dir': str(Path(cache_dir).resolve()),
        'source_dataset_dir': manifest.get('source_dataset_dir'),
        'source_format': manifest.get('source_format'),
        'reward_config': manifest.get('reward_config', default_reward_config()),
        'materialize_max_workers': manifest.get('materialize_max_workers'),
        'materialize_worker_backend': manifest.get('materialize_worker_backend'),
        'explicit_next_state_store_count': int(manifest.get('explicit_next_state_store_count', 0)),
        'has_explicit_next_state': bool(manifest.get('has_explicit_next_state', False)),
    }


def load_replay_cache_arrays(cache_dir, mmap_mode=None):
    files = replay_cache_files(cache_dir)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Replay cache is incomplete at {cache_dir}: missing {missing}')
    manifest = load_json(files['manifest'])
    arrays = {
        'states': np.load(files['states'], mmap_mode=mmap_mode),
        'actions': np.load(files['actions'], mmap_mode=mmap_mode),
        'next_states': np.load(files['next_states'], mmap_mode=mmap_mode),
        'rewards': np.load(files['rewards'], mmap_mode=mmap_mode),
        'dones': np.load(files['dones'], mmap_mode=mmap_mode),
    }
    validate_replay_cache_arrays(manifest, arrays, cache_dir)
    return manifest, arrays


def validate_replay_cache_arrays(manifest, arrays, cache_dir):
    num_transitions = int(manifest['num_transitions'])
    state_dim = int(manifest['state_dim'])
    action_dim = int(manifest['action_dim'])
    expected_shapes = {
        'states': (num_transitions, state_dim),
        'actions': (num_transitions, action_dim),
        'next_states': (num_transitions, state_dim),
        'rewards': (num_transitions, 1),
        'dones': (num_transitions, 1),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(arrays[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f'Replay cache array {name!r} at {cache_dir} has shape '
                f'{actual_shape}, expected {expected_shape}.'
            )


def load_replay_cache_into_buffer(cache_dir, buffer, state_keys=None):
    manifest, arrays = load_replay_cache_arrays(cache_dir)
    cache_state_keys = list(manifest['state_keys'])
    if state_keys is not None and list(state_keys) != cache_state_keys:
        raise ValueError(
            'Replay cache state keys do not match requested state keys: '
            f'cache={cache_state_keys!r}, requested={list(state_keys)!r}'
        )
    if arrays['states'].shape[1] != buffer.states.shape[1]:
        raise ValueError(
            f'Replay cache state_dim={arrays["states"].shape[1]} does not match '
            f'buffer state_dim={buffer.states.shape[1]}'
        )
    if arrays['actions'].shape[1] != buffer.actions.shape[1]:
        raise ValueError(
            f'Replay cache action_dim={arrays["actions"].shape[1]} does not match '
            f'buffer action_dim={buffer.actions.shape[1]}'
        )
    num_transitions = arrays['states'].shape[0]
    if num_transitions > buffer.max_size:
        raise ValueError(
            f'Replay cache has {num_transitions} transitions but buffer capacity is {buffer.max_size}'
        )

    buffer.states[:num_transitions] = arrays['states']
    buffer.actions[:num_transitions] = arrays['actions']
    buffer.next_states[:num_transitions] = arrays['next_states']
    buffer.rewards[:num_transitions] = arrays['rewards']
    buffer.dones[:num_transitions] = arrays['dones']
    buffer.ptr = num_transitions % buffer.max_size
    buffer.size = num_transitions


def describe_dataset_with_replay_cache(directory, cache_dir=None, prefer_cache=True,
                                       show_progress=False, max_workers=1,
                                       worker_backend='process'):
    cache_path = replay_cache_path(directory, cache_dir=cache_dir)
    if prefer_cache and replay_cache_exists(cache_path):
        specs = infer_replay_cache_specs(cache_path)
        description = dict(specs)
        description.update({
            'selected_format': 'replay_cache',
            'replay_shard_count': 0,
            'zarr_store_count': 0,
            'json_file_count': 0,
            'zarr_takes_precedence': False,
            'dataset_has_explicit_next_state': specs.get('has_explicit_next_state', False),
            'zarr_explicit_next_state_store_count': specs.get('explicit_next_state_store_count', 0),
            'replay_cache_dir': str(cache_path),
            'replay_cache_used': True,
        })
        return description

    description = describe_dataset(
        directory,
        show_progress=show_progress,
        max_workers=max_workers,
        worker_backend=worker_backend,
    )
    description.update({
        'replay_cache_dir': str(cache_path),
        'replay_cache_used': False,
        'reward_config': copy.deepcopy(
            load_json(dataset_paths(directory)['manifest']).get(
                'reward_config',
                default_reward_config(),
            )
        ),
    })
    return description


def _progress_iter(iterable, *, total=None, desc=None, enabled=True):
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, total=total, desc=desc, unit='transition')


def _load_zarr_transition_block(store_path, state_keys=None):
    dataset = _zarr_reward_components_dataset(store_path)
    try:
        keys = list(state_keys) if state_keys is not None else _zarr_state_keys(dataset)
        state_arrays = []
        for key in keys:
            var_name = f'state_{key}'
            if var_name not in dataset:
                raise KeyError(f'Missing Zarr state variable {var_name!r} in {store_path}')
            state_arrays.append(np.asarray(dataset[var_name].values, dtype=np.float32))

        if not state_arrays:
            state_dim = len(keys)
            action_dim = int(dataset.sizes.get('action_dim', 0))
            return {
                'state_keys': keys,
                'has_explicit_next_state': False,
                'states': np.empty((0, state_dim), dtype=np.float32),
                'actions': np.empty((0, action_dim), dtype=np.float32),
                'next_states': np.empty((0, state_dim), dtype=np.float32),
                'rewards': np.empty((0, 1), dtype=np.float32),
                'dones': np.empty((0, 1), dtype=np.float32),
            }

        states = np.ascontiguousarray(np.stack(state_arrays, axis=1), dtype=np.float32)
        has_explicit_next_state = _zarr_has_explicit_next_state(dataset, keys)
        if has_explicit_next_state:
            next_state_arrays = [
                np.asarray(dataset[f'next_state_{key}'].values, dtype=np.float32)
                for key in keys
            ]
            next_states = np.ascontiguousarray(
                np.stack(next_state_arrays, axis=1),
                dtype=np.float32,
            )
        else:
            next_states = np.zeros_like(states)
            if states.shape[0] > 1:
                dones = np.asarray(dataset['done'].values, dtype=np.float32)
                not_terminal = dones[:-1] < 0.5
                next_indices = np.nonzero(not_terminal)[0]
                next_states[next_indices] = states[next_indices + 1]

        actions = np.asarray(dataset['action'].values, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        rewards = np.asarray(dataset['reward'].values, dtype=np.float32).reshape(-1, 1)
        dones = np.asarray(dataset['done'].values, dtype=np.float32).reshape(-1, 1)

        return {
            'state_keys': keys,
            'has_explicit_next_state': has_explicit_next_state,
            'states': states,
            'actions': np.ascontiguousarray(actions, dtype=np.float32),
            'next_states': next_states,
            'rewards': np.ascontiguousarray(rewards, dtype=np.float32),
            'dones': np.ascontiguousarray(dones, dtype=np.float32),
        }
    finally:
        dataset.close()


def _fill_replay_arrays_from_transitions(directory, state_keys, arrays, *, total=None,
                                         show_progress=True):
    write_idx = 0
    transitions = _progress_iter(
        iter_d4rl_transitions(directory, state_keys),
        total=total,
        desc='materialize replay cache',
        enabled=show_progress,
    )
    for state, action, next_state, reward, done in transitions:
        arrays['states'][write_idx] = state
        arrays['actions'][write_idx] = action
        arrays['next_states'][write_idx] = next_state
        arrays['rewards'][write_idx] = reward
        arrays['dones'][write_idx] = done
        write_idx += 1
    return write_idx


def _fill_replay_arrays_from_zarr_parallel(directory, state_keys, arrays, *,
                                           total=None, max_workers=4,
                                           show_progress=True,
                                           worker_backend='process'):
    stores = find_full_trajectory_zarr_stores(directory)
    if max_workers <= 1 or not stores:
        return _fill_replay_arrays_from_transitions(
            directory,
            state_keys,
            arrays,
            total=total,
            show_progress=show_progress,
        )

    blocks = [None] * len(stores)
    executor_cls = _replay_cache_executor(worker_backend)
    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=total, desc='read zarr replay blocks', unit='transition')

    try:
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_zarr_transition_block, store_path, state_keys): idx
                for idx, store_path in enumerate(stores)
            }
            for future in as_completed(futures):
                idx = futures[future]
                block = future.result()
                blocks[idx] = block
                if show_progress:
                    progress.update(block['states'].shape[0])
    finally:
        if show_progress:
            progress.close()

    write_idx = 0
    write_iter = blocks
    if show_progress:
        from tqdm import tqdm

        write_iter = tqdm(blocks, desc='write replay cache', unit='trajectory')

    for block in write_iter:
        if block is None:
            continue
        block_size = block['states'].shape[0]
        if block_size == 0:
            continue
        end_idx = write_idx + block_size
        arrays['states'][write_idx:end_idx] = block['states']
        arrays['actions'][write_idx:end_idx] = block['actions']
        arrays['next_states'][write_idx:end_idx] = block['next_states']
        arrays['rewards'][write_idx:end_idx] = block['rewards']
        arrays['dones'][write_idx:end_idx] = block['dones']
        write_idx = end_idx
    return write_idx


def _load_zarr_replay_blocks(directory, state_keys=None, *, max_workers=4,
                             show_progress=True, worker_backend='process'):
    stores = find_full_trajectory_zarr_stores(directory)
    if not stores:
        raise FileNotFoundError(f'No trajectory_*.zarr stores found in {directory}')

    blocks = [None] * len(stores)
    executor_cls = _replay_cache_executor(worker_backend)
    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=len(stores), desc='load zarr replay blocks', unit='store')

    try:
        if max_workers > 1:
            with executor_cls(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_load_zarr_transition_block, store_path, state_keys): idx
                    for idx, store_path in enumerate(stores)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    blocks[idx] = future.result()
                    if progress is not None:
                        progress.update(1)
        else:
            for idx, store_path in enumerate(stores):
                blocks[idx] = _load_zarr_transition_block(store_path, state_keys)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return blocks


def _load_replay_shard_blocks(directory, state_keys=None, *, show_progress=True):
    shard_files = find_replay_shard_files(directory)
    if not shard_files:
        raise FileNotFoundError(f'No trajectory_*.npz replay shards found in {directory}')

    shard_iter = shard_files
    if show_progress:
        from tqdm import tqdm

        shard_iter = tqdm(shard_files, desc='load replay shards', unit='shard')
    return [
        _load_replay_shard_block(shard_path, state_keys=state_keys)
        for shard_path in shard_iter
    ]


def _replay_specs_from_zarr_blocks(blocks):
    state_keys = None
    action_dim = None
    total_transitions = 0
    nonempty_count = 0
    explicit_next_state_count = 0

    for block in blocks:
        if block is None:
            continue
        block_state_keys = list(block['state_keys'])
        block_action_dim = int(block['actions'].shape[1])
        if state_keys is None:
            state_keys = block_state_keys
            action_dim = block_action_dim
        elif state_keys != block_state_keys:
            raise ValueError(
                'Zarr replay block state keys differ within dataset: '
                f'expected={state_keys!r}, got={block_state_keys!r}'
            )
        elif action_dim != block_action_dim:
            raise ValueError(
                'Zarr replay block action_dim differs within dataset: '
                f'expected={action_dim}, got={block_action_dim}'
            )

        block_size = int(block['states'].shape[0])
        if block_size == 0:
            continue
        total_transitions += block_size
        nonempty_count += 1
        if block.get('has_explicit_next_state', False):
            explicit_next_state_count += 1

    if state_keys is None or action_dim is None:
        raise ValueError('No non-empty Zarr replay blocks found.')

    return {
        'selected_format': 'zarr',
        'num_trajectories': nonempty_count,
        'num_transitions': total_transitions,
        'state_dim': len(state_keys),
        'action_dim': action_dim,
        'state_keys': state_keys,
        'zarr_explicit_next_state_store_count': explicit_next_state_count,
        'dataset_has_explicit_next_state': explicit_next_state_count == nonempty_count,
    }


def _write_replay_arrays_from_blocks(blocks, arrays, *, show_progress=True):
    write_idx = 0
    write_iter = blocks
    if show_progress:
        from tqdm import tqdm

        write_iter = tqdm(blocks, desc='write replay cache', unit='trajectory')

    for block in write_iter:
        if block is None:
            continue
        block_size = block['states'].shape[0]
        if block_size == 0:
            continue
        end_idx = write_idx + block_size
        arrays['states'][write_idx:end_idx] = block['states']
        arrays['actions'][write_idx:end_idx] = block['actions']
        arrays['next_states'][write_idx:end_idx] = block['next_states']
        arrays['rewards'][write_idx:end_idx] = block['rewards']
        arrays['dones'][write_idx:end_idx] = block['dones']
        write_idx = end_idx
    return write_idx


def _open_replay_memmaps(files, num_transitions, state_dim, action_dim):
    return {
        'states': np.lib.format.open_memmap(
            files['states'], mode='w+', dtype=np.float32, shape=(num_transitions, state_dim)
        ),
        'actions': np.lib.format.open_memmap(
            files['actions'], mode='w+', dtype=np.float32, shape=(num_transitions, action_dim)
        ),
        'next_states': np.lib.format.open_memmap(
            files['next_states'], mode='w+', dtype=np.float32, shape=(num_transitions, state_dim)
        ),
        'rewards': np.lib.format.open_memmap(
            files['rewards'], mode='w+', dtype=np.float32, shape=(num_transitions, 1)
        ),
        'dones': np.lib.format.open_memmap(
            files['dones'], mode='w+', dtype=np.float32, shape=(num_transitions, 1)
        ),
    }


def materialize_replay_cache(dataset_dir, cache_dir=None, *, overwrite=False,
                             show_progress=True, max_workers=None,
                             worker_backend=None):
    """Materialize a compact IQL replay cache from replay-shard/JSON/Zarr outputs."""
    dataset_dir = Path(dataset_dir).resolve()
    if max_workers is None:
        max_workers = int(
            os.environ.get(
                'REPLAY_CACHE_WORKERS',
                os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1),
            )
        )
    if worker_backend is None:
        worker_backend = os.environ.get('REPLAY_CACHE_WORKER_BACKEND', 'process')
    max_workers = max(1, int(max_workers))
    final_cache_dir = replay_cache_path(dataset_dir, cache_dir=cache_dir)
    if final_cache_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f'Replay cache already exists: {final_cache_dir}. '
                'Pass overwrite=True to rebuild it.'
            )

    replay_shards = find_replay_shard_files(dataset_dir)
    zarr_stores = find_full_trajectory_zarr_stores(dataset_dir)
    blocks = None
    if replay_shards:
        blocks = _load_replay_shard_blocks(
            dataset_dir,
            show_progress=show_progress,
        )
        specs = _replay_specs_from_zarr_blocks(blocks)
        specs['selected_format'] = 'replay_shards'
    elif zarr_stores:
        blocks = _load_zarr_replay_blocks(
            dataset_dir,
            max_workers=max_workers,
            show_progress=show_progress,
            worker_backend=worker_backend,
        )
        specs = _replay_specs_from_zarr_blocks(blocks)
    else:
        specs = describe_dataset(
            dataset_dir,
            show_progress=show_progress,
            max_workers=max_workers,
            worker_backend=worker_backend,
        )
    state_keys = list(specs['state_keys'])
    num_transitions = int(specs['num_transitions'])
    state_dim = int(specs['state_dim'])
    action_dim = int(specs['action_dim'])

    tmp_dir = final_cache_dir.with_name(
        f'.{final_cache_dir.name}.tmp.{os.getpid()}'
    )
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    files = replay_cache_files(tmp_dir)

    arrays = _open_replay_memmaps(files, num_transitions, state_dim, action_dim)

    try:
        if blocks is not None:
            written = _write_replay_arrays_from_blocks(
                blocks,
                arrays,
                show_progress=show_progress,
            )
        else:
            written = _fill_replay_arrays_from_transitions(
                dataset_dir,
                state_keys,
                arrays,
                total=num_transitions,
                show_progress=show_progress,
            )
        for array in arrays.values():
            array.flush()
        del arrays
        if written != num_transitions:
            raise ValueError(
                f'Expected to write {num_transitions} transitions, wrote {written}.'
            )

        manifest = {
            'schema_version': REPLAY_CACHE_VERSION,
            'created_at': utc_now_iso(),
            'source_dataset_dir': str(dataset_dir),
            'source_format': specs['selected_format'],
            'reward_config': copy.deepcopy(load_json(dataset_paths(dataset_dir)['manifest']).get('reward_config', default_reward_config())),
            'materialize_max_workers': int(max_workers),
            'materialize_worker_backend': worker_backend,
            'num_trajectories': int(specs['num_trajectories']),
            'num_transitions': num_transitions,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'state_keys': state_keys,
            'explicit_next_state_store_count': int(
                specs.get('zarr_explicit_next_state_store_count', 0)
            ),
            'has_explicit_next_state': bool(
                specs.get('dataset_has_explicit_next_state', False)
            ),
            'files': {
                key: path.name
                for key, path in files.items()
                if key != 'manifest'
            },
        }
        atomic_write_json(files['manifest'], manifest)
        if final_cache_dir.exists():
            shutil.rmtree(final_cache_dir)
        os.replace(tmp_dir, final_cache_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    return load_replay_cache_manifest(final_cache_dir)
