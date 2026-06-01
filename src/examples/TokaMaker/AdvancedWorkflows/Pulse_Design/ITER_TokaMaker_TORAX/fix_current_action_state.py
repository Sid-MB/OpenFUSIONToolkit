#!/usr/bin/env python3
"""
Fix collected trajectory JSON observations so state heating is the current knot.

For each transition i:
  - transitions[0]["s"]["ecrh"/"nbi"] = fixed knot at t=80: (20e6, 33e6)
  - transitions[i]["s"]["ecrh"/"nbi"] = actions_raw[i - 1] for i > 0

The stored transition action "a" is left unchanged: it remains the next knot at
t_next. By default corrected files are written to OUTPUT_DIR. Use --in-place to
edit files directly; backups are written as .bak unless --no-backup is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


FIXED_T80_ECRH_W = 20.0e6
FIXED_T80_NBI_W = 33.0e6


def current_heating_for_transition(actions_raw, index):
    if index == 0:
        return FIXED_T80_ECRH_W, FIXED_T80_NBI_W
    return float(actions_raw[index - 1][0]), float(actions_raw[index - 1][1])


def fix_payload(payload, *, sync_s_next=False):
    actions_raw = payload.get("actions_raw")
    transitions = payload.get("transitions")

    if not isinstance(actions_raw, list) or not isinstance(transitions, list):
        raise ValueError("payload must contain list fields 'actions_raw' and 'transitions'")
    if len(actions_raw) < len(transitions):
        raise ValueError(
            f"actions_raw has {len(actions_raw)} rows but transitions has {len(transitions)} rows"
        )

    changed = 0
    for i, transition in enumerate(transitions):
        state = transition.get("s")
        if not isinstance(state, dict):
            raise ValueError(f"transition {i} is missing dict field 's'")

        ecrh, nbi = current_heating_for_transition(actions_raw, i)
        old = (state.get("ecrh"), state.get("nbi"))
        state["ecrh"] = ecrh
        state["nbi"] = nbi
        changed += int(old != (ecrh, nbi))

    if sync_s_next:
        for i, transition in enumerate(transitions):
            if "s_next" not in transition:
                continue
            if i < len(transitions) - 1:
                next_state = transitions[i + 1].get("s", {})
                transition["s_next"]["ecrh"] = next_state.get("ecrh")
                transition["s_next"]["nbi"] = next_state.get("nbi")
            else:
                transition["s_next"]["ecrh"] = 0.0
                transition["s_next"]["nbi"] = 0.0

    return changed


def fix_file(input_path, output_path, *, sync_s_next=False):
    with input_path.open("r") as f:
        payload = json.load(f)

    changed = fix_payload(payload, sync_s_next=sync_s_next)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp_path.replace(output_path)
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Shift trajectory state ecrh/nbi fields from future knot to current knot."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing trajectory_*.json files")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Directory for corrected JSON files. Required unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Edit files in input_dir directly.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="With --in-place, do not create .bak backups.",
    )
    parser.add_argument(
        "--sync-s-next",
        action="store_true",
        help="Also update s_next ecrh/nbi when s_next exists.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.in_place:
        output_dir = input_dir
    else:
        if args.output_dir is None:
            raise SystemExit("output_dir is required unless --in-place is used")
        output_dir = args.output_dir.resolve()

    files = sorted(input_dir.glob("trajectory_*.json"))
    if not files:
        raise SystemExit(f"No trajectory_*.json files found in {input_dir}")

    total_changed = 0
    for input_path in files:
        output_path = output_dir / input_path.name
        if args.in_place and not args.no_backup:
            backup_path = input_path.with_suffix(input_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)
        changed = fix_file(input_path, output_path, sync_s_next=args.sync_s_next)
        total_changed += changed
        print(f"{input_path.name}: updated {changed} transition states")

    print(
        f"Done. Processed {len(files)} files; updated {total_changed} transition states. "
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
