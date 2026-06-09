#!/usr/bin/env bash
# postprocess_pfusion_evals.sh — Generate heating-schedule charts from pfusion re-evals
# and copy them to different-algorithms/charts/.
#
# Run after all pfusion eval jobs complete (use --dependency=afterok:... at submission).
# Generates individual heating_schedule.{pdf,pgf} for each run, copies to charts/,
# then regenerates the grid chart with all runs including the new pfusion and pellet ones.
# Prints a metrics table to stdout for pasting into RESULTS.md.

#SBATCH --account=nlp
#SBATCH --mem=8G
#SBATCH --partition=john,sc-loprio
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/postprocess_pfusion_evals-${SLURM_JOB_ID:-$$}.out") \
  2> >(tee -a "${LOG_DIR}/postprocess_pfusion_evals-${SLURM_JOB_ID:-$$}.err" >&2)

CHARTS_DIR="${PROJECT_DIR}/different-algorithms/charts"
PFUSION_EVAL_BASE="${PROJECT_DIR}/out/iql/pfusion_eval"

# Map: chart_dir_name -> actor_eval_summary.json path
declare -A RUNS
RUNS[iql_d52e9z6h]="${PFUSION_EVAL_BASE}/d52e9z6h/actor_eval_summary.json"
RUNS[iql_to5oswfw]="${PFUSION_EVAL_BASE}/to5oswfw/actor_eval_summary.json"
RUNS[iql_ms6z7gz8]="${PFUSION_EVAL_BASE}/ms6z7gz8/actor_eval_summary.json"
RUNS[bc_d7w93xr7]="${PFUSION_EVAL_BASE}/d7w93xr7/actor_eval_summary.json"
RUNS[cql_58znl1w7]="${PFUSION_EVAL_BASE}/58znl1w7/actor_eval_summary.json"
RUNS[td3bc_py0nd5nm]="${PFUSION_EVAL_BASE}/py0nd5nm/actor_eval_summary.json"
RUNS[iql_dj4r5s8o]="${PFUSION_EVAL_BASE}/dj4r5s8o/actor_eval_summary.json"
RUNS[iql_w0gqz7fd]="${PFUSION_EVAL_BASE}/w0gqz7fd/actor_eval_summary.json"
# pfusion_standard already evaluated with pfusion reward
RUNS[pfusion_2tjm9kx1]="${PROJECT_DIR}/out/iql/reward_68ceccd06040/2tjm9kx1/actor_eval/actor_eval_summary.json"
# pellet default-reward run
RUNS[pellet_x0snq72a]="${PROJECT_DIR}/run_prev_action_pellet_full_20260603_0956/eval/checkpoint_step_19000_eval_cpu_15830880/actor_eval_summary.json"

echo "=== Generating individual heating schedule charts ==="
for chart_name in "${!RUNS[@]}"; do
  summary="${RUNS[$chart_name]}"
  if [ ! -f "$summary" ]; then
    echo "SKIP $chart_name — summary not found: $summary"
    continue
  fi
  out_dir="${CHARTS_DIR}/${chart_name}"
  mkdir -p "$out_dir"
  echo "Generating $chart_name ..."
  uv run python visualize/plot_heating_schedule.py "$out_dir" "$summary" && echo "  OK" || echo "  FAILED"
done

echo ""
echo "=== Regenerating grid chart ==="
GRID_SUMMARIES=()
# Original 6 algorithm comparison runs (pfusion re-eval)
for run_id in d52e9z6h to5oswfw ms6z7gz8 d7w93xr7 58znl1w7 py0nd5nm; do
  s="${PFUSION_EVAL_BASE}/${run_id}/actor_eval_summary.json"
  [ -f "$s" ] && GRID_SUMMARIES+=("$s")
done
# pfusion standard
s="${PROJECT_DIR}/out/iql/reward_68ceccd06040/2tjm9kx1/actor_eval/actor_eval_summary.json"
[ -f "$s" ] && GRID_SUMMARIES+=("$s")

if [ ${#GRID_SUMMARIES[@]} -gt 0 ]; then
  uv run python visualize/plot_heating_schedule.py --grid "${CHARTS_DIR}/grid" "${GRID_SUMMARIES[@]}" && echo "Grid OK" || echo "Grid FAILED"
else
  echo "No summaries available for grid"
fi

echo ""
echo "=== Generating LaTeX table ==="
uv run python visualize/generate_latex_table.py "${CHARTS_DIR}/grid" \
  --project-dir "${PROJECT_DIR}" \
  --pfusion-eval-dir "${PFUSION_EVAL_BASE}"

echo ""
echo "=== Metrics table (for RESULTS.md) ==="
uv run python - <<'PYEOF'
import json, os
from pathlib import Path

PROJECT = Path("/juice2/scr2/siddharth/OpenFUSIONToolkit/src/examples/TokaMaker/AdvancedWorkflows/Pulse_Design/ITER_TokaMaker_TORAX")
PFUSION_BASE = PROJECT / "out/iql/pfusion_eval"

RUNS = [
    ("IQL (best)",     "d52e9z6h",  PFUSION_BASE / "d52e9z6h/actor_eval_summary.json",    "pfusion re-eval; trained standard reward"),
    ("IQL (median)",   "to5oswfw",  PFUSION_BASE / "to5oswfw/actor_eval_summary.json",    "pfusion re-eval; action_rate_penalty=0.1"),
    ("IQL (worst)",    "ms6z7gz8",  PFUSION_BASE / "ms6z7gz8/actor_eval_summary.json",    "pfusion re-eval; same config as best, diff seed"),
    ("BC",             "d7w93xr7",  PFUSION_BASE / "d7w93xr7/actor_eval_summary.json",    "pfusion re-eval"),
    ("CQL",            "58znl1w7",  PFUSION_BASE / "58znl1w7/actor_eval_summary.json",    "pfusion re-eval"),
    ("TD3+BC",         "py0nd5nm",  PFUSION_BASE / "py0nd5nm/actor_eval_summary.json",    "pfusion re-eval"),
    ("IQL strong-pen", "w0gqz7fd",  PFUSION_BASE / "w0gqz7fd/actor_eval_summary.json",    "pfusion re-eval; fgw=4.0,q95=3.5"),
    ("IQL full-data",  "dj4r5s8o",  PFUSION_BASE / "dj4r5s8o/actor_eval_summary.json",   "pfusion re-eval; run_prev_action_full dataset"),
    ("pfusion_std",    "2tjm9kx1",  PROJECT / "out/iql/reward_68ceccd06040/2tjm9kx1/actor_eval/actor_eval_summary.json", "trained pfusion; 60k steps"),
    ("pellet (def.r)", "x0snq72a",  PROJECT / "run_prev_action_pellet_full_20260603_0956/eval/checkpoint_step_19000_eval_cpu_15830880/actor_eval_summary.json", "3D action; default reward; step 19k"),
]

print(f"{'Run':<18} {'ID':<12} {'reward_total':>12} {'Q_max':>7} {'Q_flat':>7} {'H98':>6} {'q95':>6}  Notes")
print("-" * 100)
for label, run_id, path, notes in RUNS:
    if not Path(path).exists():
        print(f"{label:<18} {run_id:<12} {'(pending)':>12}  -  {notes}")
        continue
    with open(path) as f:
        d = json.load(f)
    m = d.get("metrics", d)
    rt  = m.get("actor_eval/reward_total", float("nan"))
    qm  = m.get("actor_eval/Q_max", float("nan"))
    qf  = m.get("actor_eval/Q_flattop_avg", float("nan"))
    h98 = m.get("actor_eval/H98_flattop_avg", float("nan"))
    q95 = m.get("actor_eval/q95_min", float("nan"))
    print(f"{label:<18} {run_id:<12} {rt:>12.2f} {qm:>7.2f} {qf:>7.2f} {h98:>6.3f} {q95:>6.3f}  {notes}")
PYEOF
