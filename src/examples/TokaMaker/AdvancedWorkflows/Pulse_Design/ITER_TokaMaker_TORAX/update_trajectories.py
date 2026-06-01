#!/usr/bin/env python3
"""Rewrite a collected dataset's rewards from saved scalar traces.

This tool is the recovery path for future RLRewardConfig changes. It reads the
compact per-trajectory reward-recalc stats saved at collection time, recomputes
the per-decision rewards under the current reward defaults, and writes a copy
of the dataset under a reward-variant directory. The rewritten dataset keeps
the original trajectories and action sampling, but replaces the baked reward
values and rebuilds the replay cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from dataloader import (
    atomic_save_npz_once,
    atomic_write_json,
    dataset_paths,
    default_reward_config,
    ensure_dataset_dirs,
    load_json,
    load_reward_recalc_stats,
    materialize_replay_cache,
    recompute_reward_series_from_stats,
    reward_config_to_dict,
    utc_now_iso,
)


def stable_reward_variant_name(reward_config):
    blob = json.dumps(reward_config, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]
    return f'reward_{digest}'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Rewrite a collected dataset into a reward-specific variant using '
            'the saved reward-recalc stats bundles.'
        )
    )
    parser.add_argument(
        'source_dataset_dir',
        type=Path,
        help=(
            'Collected dataset root to rewrite. It must contain replay_shards '
            'and reward_recalc_stats from a run collected with '
            '--save_stats_for_reward_recalc.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help=(
            'Destination dataset root. Leave unset to create a copy under '
            '<source_dataset_dir>/reward_variants/reward_<hash>.'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help=(
            'Replace an existing reward-variant output directory. Leave this '
            'off for safety when you want to preserve older reward variants.'
        ),
    )
    parser.add_argument(
        '--no_materialize_replay_cache',
        dest='materialize_replay_cache',
        action='store_false',
        help=(
            'Skip rebuilding the compact replay cache after rewriting the '
            'replay shards. Use only if another job will materialize the cache '
            'later.'
        ),
    )
    parser.add_argument(
        '--materialize_replay_cache',
        dest='materialize_replay_cache',
        action='store_true',
        default=True,
        help=(
            'Rebuild <output_dir>/replay_cache from the rewritten replay '
            'shards. Keep this on for the normal end-to-end rewrite path.'
        ),
    )
    return parser.parse_args()


def load_reward_config_manifest(source_manifest):
    source_reward_config = copy.deepcopy(
        source_manifest.get('reward_config', default_reward_config())
    )
    current_reward_config = reward_config_to_dict(default_reward_config())
    return source_reward_config, current_reward_config


def copy_directory_if_present(source_dir, dest_dir):
    if source_dir.is_dir():
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)


def rewrite_trajectory_json(source_path, dest_path, new_rewards, new_reward_config,
                            source_reward_config, update_meta):
    payload = load_json(source_path)
    transitions = payload.get('transitions') or []
    if len(transitions) != len(new_rewards):
        raise ValueError(
            f'{source_path}: transition count {len(transitions)} does not '
            f'match recomputed rewards {len(new_rewards)}'
        )

    for idx, transition in enumerate(transitions):
        transition['r'] = float(new_rewards[idx])

    payload['transitions'] = transitions
    payload['reward_config'] = copy.deepcopy(new_reward_config)
    payload['reward_update'] = {
        'source_reward_config': copy.deepcopy(source_reward_config),
        'updated_reward_config': copy.deepcopy(new_reward_config),
        'update_meta': copy.deepcopy(update_meta),
    }
    atomic_write_json(dest_path, payload)


def rewrite_replay_shard(source_path, dest_path, new_rewards, new_reward_config,
                         source_reward_config, update_meta):
    with np.load(source_path, allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}

    if 'rewards' not in arrays:
        raise ValueError(f'{source_path}: replay shard missing rewards array')

    rewards = np.asarray(new_rewards, dtype=arrays['rewards'].dtype)
    rewards = rewards.reshape(arrays['rewards'].shape)
    arrays['rewards'] = rewards
    arrays['reward_config_json'] = np.asarray(
        json.dumps(new_reward_config, sort_keys=True),
        dtype=np.str_,
    )
    arrays['reward_update_json'] = np.asarray(
        json.dumps(
            {
                'source_reward_config': source_reward_config,
                'updated_reward_config': new_reward_config,
                'update_meta': update_meta,
            },
            sort_keys=True,
        ),
        dtype=np.str_,
    )
    atomic_save_npz_once(dest_path, **arrays)


def main():
    args = parse_args()
    source_dir = args.source_dataset_dir.resolve()
    source_paths = dataset_paths(source_dir)
    source_manifest_path = source_paths['manifest']
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f'Missing source dataset manifest: {source_manifest_path}')

    source_manifest = load_json(source_manifest_path)
    source_reward_config, current_reward_config = load_reward_config_manifest(source_manifest)
    reward_variant_name = stable_reward_variant_name(current_reward_config)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = source_paths['root'] / 'reward_variants' / reward_variant_name
    output_dir = output_dir.resolve()
    output_paths = dataset_paths(output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Reward-variant output already exists: {output_dir}. '
                'Pass --overwrite to replace it.'
            )
        shutil.rmtree(output_dir)

    ensure_dataset_dirs(output_dir)
    output_paths = dataset_paths(output_dir)

    if not source_paths['replay_shards'].is_dir():
        raise FileNotFoundError(
            f'Source dataset is missing replay shards: {source_paths["replay_shards"]}'
        )
    if not source_paths['reward_recalc_stats'].is_dir():
        raise FileNotFoundError(
            f'Source dataset is missing reward-recalc stats: {source_paths["reward_recalc_stats"]}\n'
            'The collection step must be run with save_stats_for_reward_recalc=true '
            'to update rewards without recollecting trajectories.'
        )

    rl_times = source_manifest.get('rl_times')
    if not rl_times:
        raise ValueError(
            'Source manifest is missing rl_times, so rewards cannot be '
            'recomputed safely.'
        )

    run_ids = []
    for source_shard in sorted(source_paths['replay_shards'].glob('trajectory_*.npz')):
        stem = source_shard.stem.split('_')[-1]
        run_ids.append(int(stem))

    if not run_ids:
        raise FileNotFoundError(
            f'No replay shards were found under {source_paths["replay_shards"]}'
        )

    reward_update_meta = {
        'source_dataset_dir': str(source_dir),
        'output_dir': str(output_dir),
        'reward_variant_name': reward_variant_name,
        'source_reward_config': copy.deepcopy(source_reward_config),
        'updated_reward_config': copy.deepcopy(current_reward_config),
        'source_created_at': source_manifest.get('created_at'),
    }

    # Copy the small provenance artifacts first.
    shutil.copy2(source_paths['actions'], output_paths['actions'])
    if source_paths['root'].joinpath('grid_search').is_dir():
        copy_directory_if_present(
            source_paths['root'].joinpath('grid_search'),
            output_paths['root'].joinpath('grid_search'),
        )
    if source_paths['failures'].is_dir():
        copy_directory_if_present(source_paths['failures'], output_paths['failures'])
    if source_paths['chunks'].is_dir():
        copy_directory_if_present(source_paths['chunks'], output_paths['chunks'])
    copy_directory_if_present(
        source_paths['reward_recalc_stats'],
        output_paths['reward_recalc_stats'],
    )

    updated_manifest = copy.deepcopy(source_manifest)
    updated_manifest['created_at'] = utc_now_iso()
    updated_manifest['source_created_at'] = source_manifest.get('created_at')
    updated_manifest['reward_config'] = copy.deepcopy(current_reward_config)
    updated_manifest['reward_update'] = copy.deepcopy(reward_update_meta)
    updated_manifest['variant_root'] = str(output_dir)
    atomic_write_json(output_paths['manifest'], updated_manifest)

    for run_id in run_ids:
        source_shard = source_paths['replay_shards'] / f'trajectory_{run_id:04d}.npz'
        dest_shard = output_paths['replay_shards'] / f'trajectory_{run_id:04d}.npz'
        source_stats = source_paths['reward_recalc_stats'] / f'trajectory_{run_id:04d}.npz'
        if not source_shard.is_file():
            raise FileNotFoundError(f'Missing source replay shard: {source_shard}')
        if not source_stats.is_file():
            raise FileNotFoundError(f'Missing reward-recalc stats bundle: {source_stats}')

        stats = load_reward_recalc_stats(source_stats)
        new_rewards = recompute_reward_series_from_stats(
            stats,
            rl_times=rl_times,
            reward_config=current_reward_config,
        )
        rewrite_replay_shard(
            source_shard,
            dest_shard,
            new_rewards,
            current_reward_config,
            source_reward_config,
            reward_update_meta,
        )

        source_json = source_paths['trajectories'] / f'trajectory_{run_id:04d}.json'
        if source_json.is_file():
            dest_json = output_paths['trajectories'] / f'trajectory_{run_id:04d}.json'
            rewrite_trajectory_json(
                source_json,
                dest_json,
                new_rewards,
                current_reward_config,
                source_reward_config,
                reward_update_meta,
            )

    materialized_cache = None
    if args.materialize_replay_cache:
        cache_manifest = materialize_replay_cache(
            output_dir,
            overwrite=True,
            show_progress=True,
        )
        materialized_cache = cache_manifest.get('source_format', 'replay_shards')

    result = {
        'source_dataset_dir': str(source_dir),
        'output_dir': str(output_dir),
        'reward_variant_name': reward_variant_name,
        'reward_update': reward_update_meta,
        'rewritten_trajectories': len(run_ids),
        'materialized_replay_cache': bool(args.materialize_replay_cache),
        'replay_cache_source_format': materialized_cache,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
