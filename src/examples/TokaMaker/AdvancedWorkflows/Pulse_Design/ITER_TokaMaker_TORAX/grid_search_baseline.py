import argparse
import csv
import json
import os
import re
from pathlib import Path
from datetime import datetime

from dataloader import find_trajectory_files


DEFAULT_DATASET_DIR = "rl_dataset_delta_sampling_maxloop=2_grid_51_preprocessed"
DEFAULT_OUTPUT_ROOT = Path("out/grid_search")
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
            "Dataset root containing trajectories/trajectory_*.json, or an old "
            f"flat trajectory_*.json directory. Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for baseline output files. Default: out/grid_search/<dataset-name>_<timestamp>_<slurm-job-id>",
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
    dataset_name = slugify(dataset_dir.resolve().name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = slugify(get_slurm_job_id())
    return DEFAULT_OUTPUT_ROOT / f"{dataset_name}_{timestamp}_{slurm_job_id}"


def resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    return default_output_dir(args.dataset_dir)


def load_trajectory(path):
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

    trajectory_files = find_trajectory_files(args.dataset_dir)
    if not trajectory_files:
        raise SystemExit(f"No trajectory_*.json files found in {args.dataset_dir}")

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    scored = []
    trajectories_by_path = {}
    summary_keys = set()
    for path in trajectory_files:
        trajectory = load_trajectory(path)
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
        json.dump(best_payload, f, indent=2)
        f.write("\n")

    top_k_rows = scored[: args.top_k]
    summary_payload = {
        "dataset_dir": str(args.dataset_dir),
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
        json.dump(summary_payload, f, indent=2)
        f.write("\n")

    print(f"Scanned {len(scored)} trajectories from {args.dataset_dir}")
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
