#!/usr/bin/env python3
"""Visualize collected TokaMaker/TORAX trajectory datasets.

The script reads the per-trajectory Zarr stores written by
``collect_trajectories_delta.py`` and emits static PNG plots. It is safe to run
while a Slurm array is still collecting trajectories because it only opens
already-published ``trajectory_*.zarr`` directories.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader import find_full_trajectory_zarr_stores, find_trajectory_files, load_json  # noqa: E402


DEFAULT_DATASET_GLOB = "rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr*"
SAFETY_LIMITS = {
    "q95": 3.0,
    "beta_N": 2.8,
    "fgw_n_e_line_avg": 0.85,
}


@dataclass(frozen=True)
class TrajectoryStore:
    path: Path
    run_id: int
    format: str


def natural_run_id(path: Path) -> int:
    match = re.search(r"trajectory_(\d+)\.(?:zarr|json)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse run id from {path}")
    return int(match.group(1))


def discover_latest_dataset(root: Path) -> Path:
    candidates = []
    for path in root.glob(DEFAULT_DATASET_GLOB):
        if find_full_trajectory_zarr_stores(path) or find_trajectory_files(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No dataset matching {DEFAULT_DATASET_GLOB!r} with Zarr stores under {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def list_stores(dataset_dir: Path) -> list[TrajectoryStore]:
    zarr_stores = [
        TrajectoryStore(path=Path(path), run_id=natural_run_id(Path(path)), format="zarr")
        for path in find_full_trajectory_zarr_stores(dataset_dir)
    ]
    if zarr_stores:
        return sorted(zarr_stores, key=lambda item: item.run_id)

    stores = [
        TrajectoryStore(path=Path(path), run_id=natural_run_id(Path(path)), format="json")
        for path in find_trajectory_files(dataset_dir)
    ]
    return sorted(stores, key=lambda item: item.run_id)


def select_stores(
    stores: list[TrajectoryStore],
    trajectory_ids: list[int] | None,
    max_trajectories: int,
) -> list[TrajectoryStore]:
    if trajectory_ids:
        by_id = {store.run_id: store for store in stores}
        missing = [run_id for run_id in trajectory_ids if run_id not in by_id]
        if missing:
            print(f"warning: missing trajectory ids: {missing}", file=sys.stderr)
        selected = [by_id[run_id] for run_id in trajectory_ids if run_id in by_id]
        if selected:
            return selected

    if len(stores) <= max_trajectories:
        return stores

    indices = np.linspace(0, len(stores) - 1, max_trajectories, dtype=int)
    return [stores[int(idx)] for idx in indices]


def read_root_attrs(store: TrajectoryStore) -> dict:
    if store.format == "json":
        payload = load_json(store.path)
        return {
            "run_id": int(payload.get("run_id", store.run_id)),
            "timestamp": payload.get("timestamp"),
            "actions_raw": payload.get("actions_raw"),
            "summary": payload.get("summary", {}),
        }

    root = zarr.open_group(str(store.path), mode="r")
    attrs = dict(root.attrs)
    attrs["run_id"] = int(attrs.get("run_id", store.run_id))
    return attrs


def read_reward_components(store: TrajectoryStore) -> xr.Dataset:
    if store.format == "json":
        payload = load_json(store.path)
        transitions = payload.get("transitions", [])
        if not transitions:
            raise ValueError(f"No transitions in {store.path}")

        decision_t = np.array([transition.get("t", np.nan) for transition in transitions], dtype=float)
        t_next = np.array([transition.get("t_next", np.nan) for transition in transitions], dtype=float)
        rewards = np.array([transition["r"] for transition in transitions], dtype=float)
        actions = np.array([transition["a"] for transition in transitions], dtype=float)
        dones = np.array([bool(transition.get("done", False)) for transition in transitions], dtype=bool)

        data_vars = {
            "reward": ("decision", rewards),
            "done": ("decision", dones),
            "action": (("decision", "action_dim"), actions),
        }
        state_keys = list(transitions[0].get("s", {}).keys())
        for key in state_keys:
            values = [transition.get("s", {}).get(key, np.nan) for transition in transitions]
            if all(isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)) for value in values):
                data_vars[f"state_{key}"] = ("decision", np.array(values, dtype=float))

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "decision": np.arange(len(transitions), dtype=np.int32),
                "decision_t": ("decision", decision_t),
                "t_next": ("decision", t_next),
                "action_dim": np.arange(actions.shape[1], dtype=np.int32),
            },
            attrs={"state_keys": state_keys, "source_format": "json"},
        )

    return xr.open_zarr(store.path, group="reward_components", consolidated=False)


def read_scalars(store: TrajectoryStore) -> xr.Dataset:
    if store.format == "json":
        return read_reward_components(store)
    return xr.open_zarr(store.path, group="scalars", consolidated=False)


def read_profiles(store: TrajectoryStore) -> xr.Dataset:
    if store.format == "json":
        raise ValueError("Full profile heatmaps require Zarr stores")
    return xr.open_zarr(store.path, group="profiles", consolidated=False)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def plot_summary_distributions(stores: list[TrajectoryStore], output_dir: Path) -> None:
    rows = []
    for store in stores:
        attrs = read_root_attrs(store)
        summary = attrs.get("summary", {})
        rows.append(
            {
                "run_id": store.run_id,
                "Q_flattop_avg": summary.get("Q_flattop_avg", np.nan),
                "Q_max": summary.get("Q_max", np.nan),
                "flux_consumed_Wb": summary.get("flux_consumed_Wb", np.nan),
                "q95_min": summary.get("q95_min", np.nan),
                "beta_N_max": summary.get("beta_N_max", np.nan),
                "f_GW_max": summary.get("f_GW_max", np.nan),
            }
        )

    metrics = [
        ("Q_flattop_avg", "Q flattop avg"),
        ("Q_max", "Q max"),
        ("flux_consumed_Wb", "Flux consumed [Wb]"),
        ("q95_min", "q95 min"),
        ("beta_N_max", "beta_N max"),
        ("f_GW_max", "f_GW max"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        values = np.array([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=min(40, max(8, len(values) // 4)), color="#36688d", alpha=0.86)
        ax.set_title(label)
        ax.set_ylabel("count")
        if key == "q95_min":
            ax.axvline(SAFETY_LIMITS["q95"], color="#b23a48", linestyle="--", linewidth=1.4)
        if key == "beta_N_max":
            ax.axvline(SAFETY_LIMITS["beta_N"], color="#b23a48", linestyle="--", linewidth=1.4)
        if key == "f_GW_max":
            ax.axvline(
                SAFETY_LIMITS["fgw_n_e_line_avg"],
                color="#b23a48",
                linestyle="--",
                linewidth=1.4,
            )

    fig.suptitle(f"Summary distributions across {len(rows)} trajectories", fontsize=14)
    fig.savefig(output_dir / "summary_distributions.png", dpi=180)
    plt.close(fig)
    save_json(output_dir / "summary_metrics.json", {"trajectories": rows})


def plot_reward_and_actions(stores: list[TrajectoryStore], output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(stores)))

    for store, color in zip(stores, colors):
        ds = read_reward_components(store)
        try:
            t = np.asarray(ds["decision_t"].values, dtype=float)
            reward = np.asarray(ds["reward"].values, dtype=float)
            action = np.asarray(ds["action"].values, dtype=float) / 1.0e6
            axes[0].plot(t, action[:, 0], color=color, alpha=0.85, linewidth=1.3)
            axes[1].plot(t, action[:, 1], color=color, alpha=0.85, linewidth=1.3)
            axes[2].plot(t, reward, color=color, alpha=0.85, linewidth=1.3, label=str(store.run_id))
        finally:
            ds.close()

    axes[0].set_ylabel("ECRH [MW]")
    axes[1].set_ylabel("NBI [MW]")
    axes[2].set_ylabel("reward")
    axes[2].set_xlabel("decision time [s]")
    axes[2].legend(title="run", ncols=min(4, max(1, len(stores))), fontsize=7)
    fig.suptitle("Sampled action schedules and rewards", fontsize=14)
    fig.savefig(output_dir / "sample_actions_rewards.png", dpi=180)
    plt.close(fig)


def plot_safety_panels(stores: list[TrajectoryStore], output_dir: Path) -> None:
    variables = [
        ("state_tx_q95", "q95", SAFETY_LIMITS["q95"], "below limit penalized"),
        ("state_tx_beta_N", "beta_N", SAFETY_LIMITS["beta_N"], "above limit penalized"),
        (
            "state_tx_fgw_n_e_line_avg",
            "Greenwald fraction",
            SAFETY_LIMITS["fgw_n_e_line_avg"],
            "above limit penalized",
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(stores)))

    for store, color in zip(stores, colors):
        ds = read_reward_components(store)
        try:
            t = np.asarray(ds["decision_t"].values, dtype=float)
            for ax, (var_name, label, limit, _) in zip(axes, variables):
                if var_name not in ds:
                    continue
                ax.plot(t, np.asarray(ds[var_name].values, dtype=float), color=color, alpha=0.82)
                ax.axhline(limit, color="#b23a48", linestyle="--", linewidth=1.1)
                ax.set_ylabel(label)
        finally:
            ds.close()

    for ax, (_, _, _, note) in zip(axes, variables):
        ax.text(0.995, 0.92, note, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[-1].set_xlabel("decision time [s]")
    fig.suptitle("Sampled safety state traces", fontsize=14)
    fig.savefig(output_dir / "sample_safety_traces.png", dpi=180)
    plt.close(fig)


def plot_scalar_timeseries(store: TrajectoryStore, output_dir: Path) -> None:
    if store.format == "json":
        variables = [
            ("state_tx_Q_fusion", "Q fusion", None),
            ("state_tx_q95", "q95", SAFETY_LIMITS["q95"]),
            ("state_tx_beta_N", "beta_N", SAFETY_LIMITS["beta_N"]),
            ("state_tx_fgw_n_e_line_avg", "Greenwald fraction", SAFETY_LIMITS["fgw_n_e_line_avg"]),
            ("state_tx_P_aux_total", "P aux total", None),
            ("state_tx_v_loop_lcfs", "V loop LCFS", None),
        ]
    else:
        variables = [
            ("Q_fusion", "Q fusion", None),
            ("q95", "q95", SAFETY_LIMITS["q95"]),
            ("beta_N", "beta_N", SAFETY_LIMITS["beta_N"]),
            ("fgw_n_e_line_avg", "Greenwald fraction", SAFETY_LIMITS["fgw_n_e_line_avg"]),
            ("P_aux_total", "P aux total", None),
            ("v_loop_lcfs", "V loop LCFS", None),
        ]

    ds = read_scalars(store)
    try:
        time_coord = "decision_t" if store.format == "json" else "time"
        t = np.asarray(ds[time_coord].values, dtype=float)
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
        for ax, (var_name, label, limit) in zip(axes.flat, variables):
            if var_name not in ds:
                ax.set_visible(False)
                continue
            values = np.asarray(ds[var_name].values, dtype=float)
            ax.plot(t, values, color="#2a6f68", linewidth=1.4)
            if limit is not None:
                ax.axhline(limit, color="#b23a48", linestyle="--", linewidth=1.1)
            ax.set_title(label)
        for ax in axes[-1, :]:
            ax.set_xlabel("time [s]")
        title = "Decision-state traces" if store.format == "json" else "Scalar time traces"
        fig.suptitle(f"{title} for trajectory {store.run_id}", fontsize=14)
        fig.savefig(output_dir / f"trajectory_{store.run_id:04d}_scalars.png", dpi=180)
        plt.close(fig)
    finally:
        ds.close()


def plot_profile_heatmaps(store: TrajectoryStore, output_dir: Path) -> None:
    if store.format == "json":
        return

    ds = read_profiles(store)
    try:
        variables = [name for name in ("T_e", "T_i", "n_e", "q") if name in ds]
        if not variables:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, var_name in zip(axes.flat, variables):
            data = ds[var_name]
            dims = list(data.dims)
            if "time" not in dims:
                ax.set_visible(False)
                continue
            rho_dim = next((dim for dim in dims if dim != "time"), None)
            if rho_dim is None:
                ax.set_visible(False)
                continue
            arr = data.transpose("time", rho_dim).values
            t = ds["time"].values
            rho = ds[rho_dim].values
            mesh = ax.pcolormesh(t, rho, arr.T, shading="auto", cmap="magma")
            fig.colorbar(mesh, ax=ax)
            ax.set_title(var_name)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(rho_dim)
        for ax in axes.flat[len(variables) :]:
            ax.set_visible(False)
        fig.suptitle(f"Profile heatmaps for trajectory {store.run_id}", fontsize=14)
        fig.savefig(output_dir / f"trajectory_{store.run_id:04d}_profiles.png", dpi=180)
        plt.close(fig)
    finally:
        ds.close()


def plot_summary_scatter(stores: list[TrajectoryStore], output_dir: Path) -> None:
    run_ids = []
    q_avg = []
    flux = []
    q95_min = []
    fgw_max = []
    beta_max = []
    for store in stores:
        attrs = read_root_attrs(store)
        summary = attrs.get("summary", {})
        run_ids.append(store.run_id)
        q_avg.append(summary.get("Q_flattop_avg", np.nan))
        flux.append(summary.get("flux_consumed_Wb", np.nan))
        q95_min.append(summary.get("q95_min", np.nan))
        fgw_max.append(summary.get("f_GW_max", np.nan))
        beta_max.append(summary.get("beta_N_max", np.nan))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].scatter(flux, q_avg, s=18, color="#2a6f68", alpha=0.75)
    axes[0, 0].set_xlabel("flux consumed [Wb]")
    axes[0, 0].set_ylabel("Q flattop avg")

    axes[0, 1].scatter(run_ids, q_avg, s=14, color="#36688d", alpha=0.75)
    axes[0, 1].set_xlabel("trajectory id")
    axes[0, 1].set_ylabel("Q flattop avg")

    axes[1, 0].scatter(run_ids, q95_min, s=14, color="#6a8d3f", alpha=0.75)
    axes[1, 0].axhline(SAFETY_LIMITS["q95"], color="#b23a48", linestyle="--")
    axes[1, 0].set_xlabel("trajectory id")
    axes[1, 0].set_ylabel("q95 min")

    axes[1, 1].scatter(fgw_max, beta_max, s=18, color="#8f5d7a", alpha=0.75)
    axes[1, 1].axvline(SAFETY_LIMITS["fgw_n_e_line_avg"], color="#b23a48", linestyle="--")
    axes[1, 1].axhline(SAFETY_LIMITS["beta_N"], color="#b23a48", linestyle="--")
    axes[1, 1].set_xlabel("f_GW max")
    axes[1, 1].set_ylabel("beta_N max")

    fig.suptitle("Trajectory summary scatter plots", fontsize=14)
    fig.savefig(output_dir / "summary_scatter.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset directory. Defaults to the newest full_zarr trajectory dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG/JSON outputs. Defaults to visualize/out/<dataset-name>.",
    )
    parser.add_argument(
        "--trajectory-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific trajectory ids to plot in detail.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=8,
        help="Number of trajectories to sample for detailed trace plots.",
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=0,
        help="Limit summary plots to the first N completed stores. 0 means all stores.",
    )
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip profile heatmap plots, which are the slowest outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir or discover_latest_dataset(REPO_ROOT)
    dataset_dir = dataset_dir.resolve()
    output_dir = args.output_dir or (REPO_ROOT / "visualize" / "out" / dataset_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    stores = list_stores(dataset_dir)
    if not stores:
        raise FileNotFoundError(f"No trajectory_*.zarr stores found in {dataset_dir}")

    summary_stores = stores[: args.summary_limit] if args.summary_limit else stores
    selected_stores = select_stores(stores, args.trajectory_ids, args.max_trajectories)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "format": stores[0].format,
        "n_trajectory_stores": len(stores),
        "summary_store_count": len(summary_stores),
        "selected_run_ids": [store.run_id for store in selected_stores],
    }
    save_json(output_dir / "visualization_manifest.json", manifest)

    print(json.dumps(manifest, indent=2), flush=True)
    plot_summary_distributions(summary_stores, output_dir)
    plot_summary_scatter(summary_stores, output_dir)
    plot_reward_and_actions(selected_stores, output_dir)
    plot_safety_panels(selected_stores, output_dir)

    for store in selected_stores:
        plot_scalar_timeseries(store, output_dir)
        if not args.skip_profiles:
            plot_profile_heatmaps(store, output_dir)

    print(f"Wrote plots to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
