"""
Plot ECRH/NBI heating schedules from an actor_eval_summary.json file.

Usage:
    uv run python plot_heating_schedule.py <summary_json> <output_dir>

Arguments:
    summary_json  Path to actor_eval_summary.json produced by the actor eval.
                  Contains action_records with decision_t, knot_t, ecrh_W, nbi_W.
    output_dir    Directory to write output files into. Will be created if it
                  doesn't exist. Outputs: heating_schedule.pdf, heating_schedule.pgf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2.0,
    "text.usetex": True,
})

_BASE_RL_ECRH_POWERS_W = {
    0: 0.0,
    10: 0.0,
    50: 20.0e6,
    80: 20.0e6,
    520: 10.0e6,
    540: 5.0e6,
    560: 0.0,
    600: 0.0,
}

_BASE_RL_NBI_POWERS_W = {
    0: 0.0,
    79: 0.0,
    80: 33.0e6,
    520: 10.0e6,
    540: 7.0e6,
    560: 4.0e6,
    580: 2.0e6,
    600: 0.0,
}


def _interp_schedule(d, tgrid):
    t = np.array(sorted(d.keys()), dtype=float)
    v = np.array([d[k] for k in t]) / 1e6
    return np.interp(tgrid, t, v)


def plot_heating_schedule(summary_json: Path, output_dir: Path):
    with open(summary_json) as f:
        summary = json.load(f)

    agent_decisions = summary["action_records"]

    t = np.linspace(0, 600, 2000)
    base_ecrh = _interp_schedule(_BASE_RL_ECRH_POWERS_W, t)
    base_nbi  = _interp_schedule(_BASE_RL_NBI_POWERS_W, t)
    mask = (t >= 80) & (t <= 520)
    base_ecrh[mask] = np.interp(t[mask], [80, 520], [20.0, 10.0])
    base_nbi[mask]  = np.interp(t[mask], [80, 520], [33.0, 10.0])

    agent_t    = np.array([d["knot_t"]  for d in agent_decisions])
    agent_ecrh = np.array([d["ecrh_W"]  for d in agent_decisions]) / 1e6
    agent_nbi  = np.array([d["nbi_W"]   for d in agent_decisions]) / 1e6

    agent_t_full    = np.concatenate([[80], agent_t,    [520]])
    agent_ecrh_full = np.concatenate([[20.0], agent_ecrh, [10.0]])
    agent_nbi_full  = np.concatenate([[33.0], agent_nbi,  [10.0]])

    idx = np.argsort(agent_t_full)
    agent_t_full    = agent_t_full[idx]
    agent_ecrh_full = agent_ecrh_full[idx]
    agent_nbi_full  = agent_nbi_full[idx]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.axvspan(0,   80,  color="#ffcccc", alpha=0.4)
    ax.axvspan(80,  520, color="#ccffcc", alpha=0.4)
    ax.axvspan(520, 600, color="#ffcccc", alpha=0.4)

    ax.plot(t, base_ecrh, "--", color="#4477cc", label="ECRH baseline")
    ax.plot(t, base_nbi,  "--", color="#ee8833", label="NBI baseline")
    ax.plot(agent_t_full, agent_ecrh_full, "-o", color="#4477cc", lw=3, ms=7, label="ECRH agent")
    ax.plot(agent_t_full, agent_nbi_full,  "-o", color="#ee8833", lw=3, ms=7, label="NBI agent")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Baseline vs Agent ECRH/NBI Schedule")
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.3)

    fixed_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="Fixed regions")
    agent_patch = mpatches.Patch(color="#ccffcc", alpha=0.6, label="Agent-controlled")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [fixed_patch, agent_patch], loc="upper right", frameon=True, fontsize=16, title="Legend", title_fontsize=16)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "heating_schedule.pdf")
    fig.savefig(output_dir / "heating_schedule.pgf")
    print(f"Saved to {output_dir}/heating_schedule.{{pdf,pgf}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_json", type=Path, help="Path to actor_eval_summary.json from the actor eval run.")
    parser.add_argument("output_dir", type=Path, help="Directory to write heating_schedule.pdf and heating_schedule.pgf into.")
    args = parser.parse_args()
    plot_heating_schedule(args.summary_json, args.output_dir)


if __name__ == "__main__":
    main()
