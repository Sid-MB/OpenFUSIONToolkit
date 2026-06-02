import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from dataloader import (
    find_full_trajectory_zarr_stores,
    find_replay_shard_files,
    find_trajectory_files,
)


DEFAULT_DATASET_DIR = "rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed"
RANKING_METRIC = "return_sum"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank simulated TokaMaker/TORAX trajectories as a best-observed grid-search baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help=(
            "Dataset root containing replay_shards/trajectory_*.npz, "
            "trajectories/trajectory_*.json, full_trajectories/trajectory_*.zarr, "
            f"or an old flat trajectory directory. Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for baseline output files. Default: <dataset-dir>/grid_search",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used for the auxiliary discounted return. Default: 0.99",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top trajectories to include in the summary JSON. Default: 20",
    )
    return parser.parse_args()


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "-", value.strip())
    slug = slug.strip("-")
    return slug or "dataset"


def get_slurm_job_id():
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID") or "no-slurm"


def default_output_dir(dataset_dir):
    return dataset_dir / "grid_search"


def resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    return default_output_dir(args.dataset_dir)


def to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def find_trajectory_inputs(dataset_dir):
    replay_shards = find_replay_shard_files(dataset_dir)
    if replay_shards:
        return "replay_shards", replay_shards

    json_files = find_trajectory_files(dataset_dir)
    if json_files:
        return "json", json_files

    zarr_stores = find_full_trajectory_zarr_stores(dataset_dir)
    if zarr_stores:
        return "zarr", zarr_stores

    return None, []


def load_replay_shard_trajectory(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            rewards = [float(value) for value in np.asarray(data["rewards"]).reshape(-1)]
            actions = to_jsonable(np.asarray(data["actions"]).tolist())
            run_id = int(np.asarray(data["run_id"]).item()) if "run_id" in data else None
            summary_json = str(np.asarray(data["summary_json"]).item()) if "summary_json" in data else "{}"
            actions_raw_json = (
                str(np.asarray(data["actions_raw_json"]).item())
                if "actions_raw_json" in data
                else None
            )
    except Exception as exc:
        raise ValueError(f"{path}: failed to open replay shard: {exc}") from exc

    try:
        summary = json.loads(summary_json) if summary_json else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed summary_json: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: summary_json must decode to an object")

    if actions_raw_json:
        try:
            actions_raw = json.loads(actions_raw_json)
        except json.JSONDecodeError:
            actions_raw = None
        if actions_raw is not None:
            actions = actions_raw

    return {
        "path": path,
        "run_id": run_id,
        "rewards": rewards,
        "actions": actions,
        "intervals": [{"t": None, "t_next": None} for _ in rewards],
        "summary": summary,
    }


def load_json_trajectory(path):
    try:
        with path.open("r") as f:
            trajectory = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc

    transitions = trajectory.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError(f"{path}: missing non-empty 'transitions' list")

    rewards = []
    actions = []
    intervals = []
    for idx, transition in enumerate(transitions):
        if "r" not in transition:
            raise ValueError(f"{path}: transition {idx} missing reward key 'r'")
        try:
            reward = float(transition["r"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: transition {idx} has non-numeric reward {transition['r']!r}") from exc

        rewards.append(reward)
        actions.append(transition.get("a"))
        intervals.append({"t": transition.get("t"), "t_next": transition.get("t_next")})

    summary = trajectory.get("summary") or {}
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: 'summary' must be an object when present")

    return {
        "path": path,
        "run_id": trajectory.get("run_id"),
        "rewards": rewards,
        "actions": actions,
        "intervals": intervals,
        "summary": summary,
    }


def load_zarr_trajectory(path):
    import xarray as xr
    import zarr

    try:
        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        dataset = xr.open_zarr(path, group="reward_components", consolidated=False)
    except Exception as exc:
        raise ValueError(f"{path}: failed to open Zarr trajectory: {exc}") from exc

    try:
        if "reward" not in dataset:
            raise ValueError(f"{path}: reward_components missing 'reward'")
        if "action" not in dataset:
            raise ValueError(f"{path}: reward_components missing 'action'")

        rewards = [float(value) for value in dataset["reward"].values]
        actions = to_jsonable(dataset["action"].values.tolist())

        decision_t = dataset.coords.get("decision_t")
        t_next = dataset.coords.get("t_next")
        if decision_t is None:
            t_values = [None] * len(rewards)
        else:
            t_values = to_jsonable(decision_t.values.tolist())
        if t_next is None:
            t_next_values = [None] * len(rewards)
        else:
            t_next_values = to_jsonable(t_next.values.tolist())

        intervals = [
            {"t": t_value, "t_next": t_next_value}
            for t_value, t_next_value in zip(t_values, t_next_values)
        ]

        summary = attrs.get("summary") or {}
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: root attr 'summary' must be an object when present")

        return {
            "path": path,
            "run_id": attrs.get("run_id"),
            "rewards": rewards,
            "actions": actions,
            "intervals": intervals,
            "summary": to_jsonable(summary),
        }
    finally:
        dataset.close()


def load_trajectory(path, input_format):
    if input_format == "replay_shards":
        return load_replay_shard_trajectory(path)
    if input_format == "json":
        return load_json_trajectory(path)
    if input_format == "zarr":
        return load_zarr_trajectory(path)
    raise ValueError(f"Unknown trajectory input format: {input_format}")


def score_trajectory(trajectory, gamma):
    rewards = trajectory["rewards"]
    return_sum = sum(rewards)
    return_discounted = sum((gamma ** idx) * reward for idx, reward in enumerate(rewards))
    reward_mean = return_sum / len(rewards)

    row = {
        "path": str(trajectory["path"]),
        "run_id": trajectory["run_id"],
        "num_transitions": len(rewards),
        "return_sum": return_sum,
        "return_discounted": return_discounted,
        "terminal_reward": rewards[-1],
        "reward_mean": reward_mean,
    }
    row.update(trajectory["summary"])
    return row


def format_best_trajectory(best_trajectory, best_row, rank):
    return {
        "rank": rank,
        "path": best_row["path"],
        "run_id": best_row["run_id"],
        "ranking_metric": RANKING_METRIC,
        "return_sum": best_row["return_sum"],
        "return_discounted": best_row["return_discounted"],
        "num_transitions": best_row["num_transitions"],
        "terminal_reward": best_row["terminal_reward"],
        "reward_mean": best_row["reward_mean"],
        "summary": best_trajectory["summary"],
        "actions": best_trajectory["actions"],
        "rewards": best_trajectory["rewards"],
        "intervals": best_trajectory["intervals"],
    }


def write_leaderboard(path, rows, summary_keys):
    base_fields = [
        "rank",
        "path",
        "run_id",
        "num_transitions",
        "return_sum",
        "return_discounted",
        "terminal_reward",
        "reward_mean",
    ]
    fieldnames = base_fields + summary_keys

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args)
    print(f"Output directory: {output_dir}")

    input_format, trajectory_files = find_trajectory_inputs(args.dataset_dir)
    if not trajectory_files:
        raise SystemExit(
            f"No trajectory_*.json files or trajectory_*.zarr stores found in {args.dataset_dir}"
        )

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    scored = []
    trajectories_by_path = {}
    summary_keys = set()
    for path in trajectory_files:
        trajectory = load_trajectory(path, input_format)
        row = score_trajectory(trajectory, args.gamma)
        trajectories_by_path[str(path)] = trajectory
        summary_keys.update(trajectory["summary"].keys())
        scored.append(row)

    scored.sort(key=lambda row: row[RANKING_METRIC], reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "grid_search_leaderboard.csv"
    best_path = output_dir / "best_trajectory.json"
    summary_path = output_dir / "grid_search_summary.json"

    write_leaderboard(leaderboard_path, scored, sorted(summary_keys))

    best_row = scored[0]
    best_trajectory = trajectories_by_path[best_row["path"]]
    best_payload = format_best_trajectory(best_trajectory, best_row, rank=1)
    with best_path.open("w") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    top_k_rows = scored[: args.top_k]
    summary_payload = {
        "dataset_dir": str(args.dataset_dir),
        "input_format": input_format,
        "num_trajectories": len(scored),
        "ranking_metric": RANKING_METRIC,
        "gamma": args.gamma,
        "top_k": args.top_k,
        "best": {
            "rank": 1,
            "path": best_row["path"],
            "run_id": best_row["run_id"],
            "return_sum": best_row["return_sum"],
            "return_discounted": best_row["return_discounted"],
            "Q_flattop_avg": best_row.get("Q_flattop_avg"),
            "flux_consumed_Wb": best_row.get("flux_consumed_Wb"),
        },
        "top_trajectories": [
            {
                "rank": row["rank"],
                "path": row["path"],
                "run_id": row["run_id"],
                "return_sum": row["return_sum"],
                "return_discounted": row["return_discounted"],
                "Q_flattop_avg": row.get("Q_flattop_avg"),
                "flux_consumed_Wb": row.get("flux_consumed_Wb"),
            }
            for row in top_k_rows
        ],
        "outputs": {
            "leaderboard_csv": str(leaderboard_path),
            "best_trajectory_json": str(best_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Scanned {len(scored)} {input_format} trajectories from {args.dataset_dir}")
    print(f"Best path: {best_row['path']}")
    print(f"Best run_id: {best_row['run_id']}")
    print(f"return_sum: {best_row['return_sum']:.6f}")
    print(f"return_discounted: {best_row['return_discounted']:.6f}")
    print(f"Q_flattop_avg: {best_row.get('Q_flattop_avg')}")
    print(f"flux_consumed_Wb: {best_row.get('flux_consumed_Wb')}")
    print(f"Wrote {leaderboard_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
