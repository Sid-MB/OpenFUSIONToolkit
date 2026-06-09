#!/usr/bin/env python3
"""Generate a LaTeX results table from actor_eval_summary.json files.

Writes a booktabs-style tabular environment to <output_dir>/table.tex.
Import in your LaTeX document with \input{table.tex} (requires \usepackage{booktabs}).

Usage:
    uv run python generate_latex_table.py <output_dir> [--project-dir DIR] [--pfusion-eval-dir DIR]

Arguments:
    output_dir          Directory to write table.tex into (created if needed). Typically
                        different-algorithms/charts/grid/.
    --project-dir DIR   Root of the ITER_TokaMaker_TORAX project. Defaults to the directory
                        containing this script.
    --pfusion-eval-dir DIR
                        Directory containing pfusion re-eval subdirs (one per run ID).
                        Defaults to <project-dir>/out/iql/pfusion_eval.

The table rows and their summary.json paths are hardcoded below and should be updated
whenever new runs are added to the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("metrics", d)


def build_table(project: Path, pfusion_base: Path) -> list[tuple[str, Path]]:
    return [
        (r"IQL (pfusion reward, 60k steps)", project / "out/iql/reward_68ceccd06040/2tjm9kx1/actor_eval/actor_eval_summary.json"),
        (r"IQL (best seed)",                 pfusion_base / "d52e9z6h/actor_eval_summary.json"),
        (r"IQL (median seed)",               pfusion_base / "to5oswfw/actor_eval_summary.json"),
        (r"IQL (worst seed)",                pfusion_base / "ms6z7gz8/actor_eval_summary.json"),
        (r"BC",                              pfusion_base / "d7w93xr7/actor_eval_summary.json"),
        (r"CQL",                             pfusion_base / "58znl1w7/actor_eval_summary.json"),
        (r"TD3+BC",                          pfusion_base / "py0nd5nm/actor_eval_summary.json"),
    ]


def generate(output_dir: Path, project: Path, pfusion_base: Path) -> Path:
    rows = build_table(project, pfusion_base)

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Reward & Peak $Q_f$ \\",
        r"\midrule",
    ]
    for name, path in rows:
        m = load_metrics(path)
        if m is None:
            lines.append(rf"{name} & -- & -- \\")
        else:
            reward = m.get("actor_eval/reward_total", float("nan"))
            qmax   = m.get("actor_eval/Q_max", float("nan"))
            lines.append(rf"{name} & {reward:.1f} & {qmax:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "table.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Directory to write table.tex into.")
    parser.add_argument("--project-dir", type=Path, default=None, help="Root of the ITER_TokaMaker_TORAX project. Defaults to the directory containing this script.")
    parser.add_argument("--pfusion-eval-dir", type=Path, default=None, help="Directory containing pfusion re-eval subdirs. Defaults to <project-dir>/out/iql/pfusion_eval.")
    args = parser.parse_args()

    project = args.project_dir or Path(__file__).resolve().parent
    pfusion_base = args.pfusion_eval_dir or (project / "out/iql/pfusion_eval")

    out_path = generate(args.output_dir, project, pfusion_base)
    print(f"Written {out_path}")
    print(out_path.read_text())


if __name__ == "__main__":
    main()
