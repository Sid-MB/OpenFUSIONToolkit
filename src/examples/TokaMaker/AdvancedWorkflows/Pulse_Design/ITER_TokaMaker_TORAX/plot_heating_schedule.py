"""
Plot ECRH/NBI heating schedules from actor_eval_summary.json files.

Single-run mode:
    uv run python plot_heating_schedule.py <summary_json> <output_dir>
    Writes heating_schedule.pdf and heating_schedule.pgf to output_dir.

Grid mode (multiple runs, paper-ready figure with unified legend):
    uv run python plot_heating_schedule.py --grid <output_dir> <summary_json> [<summary_json> ...]
    Writes heating_schedule_grid.pdf and heating_schedule_grid.pgf to output_dir.
    Layout is automatically chosen (e.g. 2x3 for 6 inputs). Each panel gets its own
    title; one unified legend is placed outside the grid.

Arguments:
    summary_json   Path(s) to actor_eval_summary.json produced by the actor eval.
                   Contains action_records with decision_t, knot_t, ecrh_W, nbi_W.
    output_dir     Directory to write output files into. Created if it doesn't exist.
    --grid         Produce a multi-panel grid figure instead of a single plot.
                   When set, output_dir must come before the list of summary JSONs.
    --ncols N      Number of columns in the grid (default: auto).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{mathptmx}",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 20,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.0,
    "text.usetex": True,
})

_BASE_RL_ECRH_POWERS_W = {
    0: 0.0, 10: 0.0, 50: 20.0e6, 80: 20.0e6,
    520: 10.0e6, 540: 5.0e6, 560: 0.0, 600: 0.0,
}
_BASE_RL_NBI_POWERS_W = {
    0: 0.0, 79: 0.0, 80: 33.0e6,
    520: 10.0e6, 540: 7.0e6, 560: 4.0e6, 580: 2.0e6, 600: 0.0,
}

_T = np.linspace(0, 600, 2000)
_MASK = (_T >= 80) & (_T <= 520)


def _interp_schedule(d, tgrid):
    t = np.array(sorted(d.keys()), dtype=float)
    v = np.array([d[k] for k in t]) / 1e6
    return np.interp(tgrid, t, v)


def _base_curves():
    base_ecrh = _interp_schedule(_BASE_RL_ECRH_POWERS_W, _T)
    base_nbi  = _interp_schedule(_BASE_RL_NBI_POWERS_W,  _T)
    base_ecrh[_MASK] = np.interp(_T[_MASK], [80, 520], [20.0, 10.0])
    base_nbi[_MASK]  = np.interp(_T[_MASK], [80, 520], [33.0, 10.0])
    return base_ecrh, base_nbi


def _load_summary(summary_json: Path):
    with open(summary_json) as f:
        summary = json.load(f)
    decisions = summary["action_records"]
    reward_total = summary.get("reward_total") or summary.get("metrics", {}).get("actor_eval/reward_total")

    algorithm = "IQL"
    checkpoint = summary.get("actor_checkpoint") or summary.get("eval_actor_checkpoint")
    if checkpoint:
        config_path = Path(checkpoint).parent / "iql_config.json"
        if config_path.exists():
            with open(config_path) as f:
                alg_raw = json.load(f).get("algorithm") or "iql"
            algorithm = alg_raw.upper()

    agent_t    = np.array([d["knot_t"] for d in decisions])
    agent_ecrh = np.array([d["ecrh_W"] for d in decisions]) / 1e6
    agent_nbi  = np.array([d["nbi_W"]  for d in decisions]) / 1e6

    t_full    = np.concatenate([[80], agent_t,    [520]])
    ecrh_full = np.concatenate([[20.0], agent_ecrh, [10.0]])
    nbi_full  = np.concatenate([[33.0], agent_nbi,  [10.0]])
    idx = np.argsort(t_full)
    return algorithm, reward_total, t_full[idx], ecrh_full[idx], nbi_full[idx]


def _draw_panel(ax, algorithm, reward_total, agent_t, agent_ecrh, agent_nbi, base_ecrh, base_nbi, show_xlabel=True, show_ylabel=True):
    ax.axvspan(0,   80,  color="#ffcccc", alpha=0.4)
    ax.axvspan(80,  520, color="#ccffcc", alpha=0.4)
    ax.axvspan(520, 600, color="#ffcccc", alpha=0.4)

    l_ecrh_base, = ax.plot(_T, base_ecrh, "--", color="#4477cc", alpha=0.35, label="ECRH baseline")
    l_nbi_base,  = ax.plot(_T, base_nbi,  "--", color="#ee8833", alpha=0.35, label="NBI baseline")
    l_ecrh,      = ax.plot(agent_t, agent_ecrh, "-o", color="#4477cc", lw=2, ms=5, label="ECRH agent")
    l_nbi,       = ax.plot(agent_t, agent_nbi,  "-o", color="#ee8833", lw=2, ms=5, label="NBI agent")

    title = f"{algorithm} Agent vs.\\ Baseline"
    if reward_total is not None:
        title += f" (reward={reward_total:.2f})"
    ax.set_title(title)
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.3)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Power (MW)")

    return l_ecrh, l_nbi, l_ecrh_base, l_nbi_base


def plot_heating_schedule(summary_json: Path, output_dir: Path):
    algorithm, reward_total, agent_t, agent_ecrh, agent_nbi = _load_summary(summary_json)
    base_ecrh, base_nbi = _base_curves()

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_panel(ax, algorithm, reward_total, agent_t, agent_ecrh, agent_nbi, base_ecrh, base_nbi)

    fixed_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="Fixed regions")
    agent_patch = mpatches.Patch(color="#ccffcc", alpha=0.6, label="Agent-controlled")
    handles, labels = ax.get_legend_handles_labels()
    agent_handles = [(h, l) for h, l in zip(handles, labels) if "agent" in l]
    base_handles  = [(h, l) for h, l in zip(handles, labels) if "baseline" in l]
    ordered = agent_handles + base_handles
    ax.legend(
        [h for h, _ in ordered] + [fixed_patch, agent_patch],
        [l for _, l in ordered] + ["Fixed regions", "Agent-controlled"],
        loc="upper right", frameon=True, title="Legend", title_fontsize=14,
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "heating_schedule.pdf")
    fig.savefig(output_dir / "heating_schedule.pgf")
    plt.close(fig)
    print(f"Saved to {output_dir}/heating_schedule.{{pdf,pgf}}")


def plot_heating_schedule_grid(summary_jsons: list, output_dir: Path, ncols: int = None):
    n = len(summary_jsons)
    if ncols is None:
        ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    base_ecrh, base_nbi = _base_curves()
    panel_w, panel_h = 7, 3.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows + 1.2), squeeze=False)

    legend_handles = None
    for i, summary_json in enumerate(summary_jsons):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        algorithm, reward_total, agent_t, agent_ecrh, agent_nbi = _load_summary(Path(summary_json))
        handles = _draw_panel(
            ax, algorithm, reward_total, agent_t, agent_ecrh, agent_nbi, base_ecrh, base_nbi,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )
        if legend_handles is None:
            legend_handles = handles  # l_ecrh, l_nbi, l_ecrh_base, l_nbi_base

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    # Unified legend below the grid
    fixed_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="Fixed regions")
    agent_patch = mpatches.Patch(color="#ccffcc", alpha=0.6, label="Agent-controlled")
    l_ecrh, l_nbi, l_ecrh_base, l_nbi_base = legend_handles
    fig.legend(
        [l_ecrh, l_nbi, l_ecrh_base, l_nbi_base, fixed_patch, agent_patch],
        ["ECRH agent", "NBI agent", "ECRH baseline", "NBI baseline", "Fixed regions", "Agent-controlled"],
        loc="lower center",
        ncol=6,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "heating_schedule_grid.pdf")
    fig.savefig(output_dir / "heating_schedule_grid.pgf")
    plt.close(fig)
    print(f"Saved to {output_dir}/heating_schedule_grid.{{pdf,pgf}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", action="store_true", help="Produce a multi-panel grid figure with a unified legend instead of a single plot.")
    parser.add_argument("--ncols", type=int, default=None, help="Number of columns in the grid (default: min(3, n)).")
    parser.add_argument("output_dir", type=Path, help="Directory to write output files into.")
    parser.add_argument("summary_jsons", type=Path, nargs="+", help="Path(s) to actor_eval_summary.json. Pass one for single mode, multiple for grid mode.")
    args = parser.parse_args()

    if args.grid or len(args.summary_jsons) > 1:
        plot_heating_schedule_grid(args.summary_jsons, args.output_dir, ncols=args.ncols)
    else:
        plot_heating_schedule(args.summary_jsons[0], args.output_dir)


if __name__ == "__main__":
    main()
