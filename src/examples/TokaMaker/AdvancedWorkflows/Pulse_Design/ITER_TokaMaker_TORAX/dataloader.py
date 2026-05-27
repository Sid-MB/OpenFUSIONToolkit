import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATASET_SCHEMA_VERSION = 1
MANIFEST_FILENAME = 'run_manifest.json'
ACTIONS_FILENAME = 'all_actions.npy'
TRAJECTORIES_DIRNAME = 'trajectories'
FAILURES_DIRNAME = 'failures'
CHUNKS_DIRNAME = 'chunks'


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
        'failures': root / FAILURES_DIRNAME,
        'chunks': root / CHUNKS_DIRNAME,
    }


def ensure_dataset_dirs(dataset_dir):
    paths = dataset_paths(dataset_dir)
    paths['root'].mkdir(parents=True, exist_ok=True)
    paths['trajectories'].mkdir(parents=True, exist_ok=True)
    paths['failures'].mkdir(parents=True, exist_ok=True)
    paths['chunks'].mkdir(parents=True, exist_ok=True)
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


def load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def create_run_manifest(*, n_trajectories, seed, max_loop, grid_size,
                        decision_times, rl_times, action_bounds,
                        sampler, start_idx=None, end_idx=None):
    return {
        'schema_version': DATASET_SCHEMA_VERSION,
        'created_at': utc_now_iso(),
        'n_trajectories': int(n_trajectories),
        'seed': int(seed),
        'max_loop': int(max_loop),
        'grid_size': int(grid_size),
        'decision_times': [int(t) for t in decision_times],
        'rl_times': [int(t) for t in rl_times],
        'action_bounds': copy.deepcopy(action_bounds),
        'sampler': copy.deepcopy(sampler),
        'requested_range': {
            'start_idx': None if start_idx is None else int(start_idx),
            'end_idx': None if end_idx is None else int(end_idx),
        },
        'layout': {
            'actions': ACTIONS_FILENAME,
            'trajectories': TRAJECTORIES_DIRNAME,
            'failures': FAILURES_DIRNAME,
            'chunks': CHUNKS_DIRNAME,
        },
    }


def manifest_comparison_subset(manifest):
    keys = (
        'schema_version',
        'n_trajectories',
        'seed',
        'max_loop',
        'grid_size',
        'decision_times',
        'rl_times',
        'action_bounds',
        'sampler',
    )
    return {key: manifest.get(key) for key in keys}


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


def failure_path(dataset_dir, run_id):
    return dataset_paths(dataset_dir)['failures'] / f'failed_run_{int(run_id):04d}.json'


def save_trajectory_atomic(dataset_dir, payload):
    run_id = int(payload['run_id'])
    path = trajectory_path(dataset_dir, run_id)
    atomic_write_json_once(path, payload)
    return str(path)


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


def load_state(state, state_keys=None):
    keys = state_keys if state_keys is not None else state.keys()
    return np.array([state[key] for key in keys], dtype=np.float32)


def infer_dataset_specs(directory):
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
    }


def iter_d4rl_transitions(directory, state_keys=None):
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


def load_d4rl_dataset(directory, buffer, state_keys=None):
    for state, action, next_state, reward, done in iter_d4rl_transitions(directory, state_keys):
        buffer.add(state, action, next_state, reward, done)
