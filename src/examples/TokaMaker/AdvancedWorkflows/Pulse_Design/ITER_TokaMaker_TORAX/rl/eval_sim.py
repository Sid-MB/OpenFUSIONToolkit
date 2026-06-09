"""Simulation-only actor evaluation.

This module owns the expensive closed-loop `tmtx.fly()` call and writes a compact
JSON summary plus the raw artifacts needed for offline postprocessing.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
import warnings
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from IQL import ReplayBuffer, normalize_buffer
from dataloader import describe_dataset_with_replay_cache, load_d4rl_dataset, reward_config_to_dict
from log import get_logger

logger = get_logger(__name__)

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")


def _jax_cache_workaround_note() -> str:
    """Return a short hint for catchable eval failures.

    This only helps for Python exceptions. Native SIGSEGV / abort failures
    inside TORAX/JAX/OFT cannot be intercepted here; they need a fresh process
    repro and, when relevant, `OFT_DISABLE_JAX_COMPILE_CACHE=1`.
    """
    return (
        "Known workaround: retry with OFT_DISABLE_JAX_COMPILE_CACHE=1 to bypass "
        "the persistent JAX cache. This is only a hint for catchable Python "
        "exceptions; native segfaults still bypass Python exception handling."
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _load_checkpoint_reward_config(actor_checkpoint, ckpt):
    """Return the reward config recorded with the checkpoint, if available.

    We prefer the config saved inside the checkpoint itself. Older checkpoints
    may only have a sibling `iql_config.json`, so we fall back to that. If no
    provenance exists, return ``None`` and let the caller fail with a helpful
    mismatch message unless the user explicitly allows mismatched rewards.
    """
    checkpoint_config = ckpt.get("config")
    if isinstance(checkpoint_config, dict):
        for key in ("dataset_reward_config", "reward_config"):
            if checkpoint_config.get(key) is not None:
                return reward_config_to_dict(checkpoint_config.get(key)), f"checkpoint config field {key}"

    # Search for iql_config.json in the checkpoint's directory and one level up
    # (intermediate checkpoints live in a checkpoints/ subdirectory).
    ckpt_dir = Path(actor_checkpoint).resolve().parent
    for search_dir in (ckpt_dir, ckpt_dir.parent):
        config_path = search_dir / "iql_config.json"
        if config_path.is_file():
            try:
                with config_path.open() as handle:
                    saved_config = json.load(handle)
                if isinstance(saved_config, dict):
                    for key in ("dataset_reward_config", "reward_config"):
                        if saved_config.get(key) is not None:
                            return reward_config_to_dict(saved_config.get(key)), f"{config_path} field {key}"
            except Exception as exc:
                logger.warning("Could not read %s for reward provenance: %s", config_path, exc)
    return None, None


def _reward_config_mismatch_message(*, actor_checkpoint, train_reward_config, train_reward_source,
                                    eval_reward_config, allow_mismatched_rewards):
    train_text = json.dumps(train_reward_config, sort_keys=True, indent=2) if train_reward_config is not None else "<unavailable>"
    eval_text = json.dumps(eval_reward_config, sort_keys=True, indent=2) if eval_reward_config is not None else "<unavailable>"
    return (
        "Reward configuration mismatch detected for actor eval.\n"
        f"  checkpoint: {actor_checkpoint}\n"
        f"  checkpoint reward provenance: {train_reward_source or 'unavailable'}\n"
        f"  allow_mismatched_rewards: {bool(allow_mismatched_rewards)}\n"
        "\n"
        "The checkpoint was trained with:\n"
        f"{train_text}\n"
        "\n"
        "The current eval runtime will use:\n"
        f"{eval_text}\n"
        "\n"
        "If you intentionally want to evaluate across a reward change, rerun with "
        "`ALLOW_MISMATCHED_REWARDS=1` or pass `--allow_mismatched_rewards` to the eval CLI."
    )


def _check_reward_config_match(*, actor_checkpoint, train_reward_config, train_reward_source,
                               eval_reward_config, allow_mismatched_rewards):
    train_cfg = reward_config_to_dict(train_reward_config) if train_reward_config is not None else None
    eval_cfg = reward_config_to_dict(eval_reward_config) if eval_reward_config is not None else None
    mismatch = train_cfg is None or eval_cfg is None or train_cfg != eval_cfg
    if not mismatch:
        return True
    message = _reward_config_mismatch_message(
        actor_checkpoint=actor_checkpoint,
        train_reward_config=train_cfg,
        train_reward_source=train_reward_source,
        eval_reward_config=eval_cfg,
        allow_mismatched_rewards=allow_mismatched_rewards,
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    if not allow_mismatched_rewards:
        raise RuntimeError(message)
    logger.warning("Proceeding despite reward mismatch because allow_mismatched_rewards is enabled.")
    return False


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _bundle_safe(value):
    """Return a pickle-friendly snapshot of nested eval artifacts.

    We keep numeric / array-like payloads intact and drop opaque runtime
    objects that TORAX or ctypes backends attach to the live `tmtx`.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            item_safe = _bundle_safe(item)
            if item_safe is not None:
                safe[str(key)] = item_safe
        return safe
    if isinstance(value, (list, tuple)):
        safe_list = []
        for item in value:
            item_safe = _bundle_safe(item)
            if item_safe is not None:
                safe_list.append(item_safe)
        return safe_list

    # Preserve a few simple containers that are already picklable.
    if isinstance(value, (set, frozenset)):
        return [_bundle_safe(item) for item in value if _bundle_safe(item) is not None]

    try:
        pickle.dumps(value)
    except Exception:
        return None
    return value


def _action_history_to_array(actions_history):
    """Normalize RL action history into a numeric array and structured records.

    Returns a (n_decisions, action_dim) array where action_dim is 2 for ECRH+NBI
    actors and 3 for ECRH+NBI+pellet actors. All rows have the same width; rows
    from 2-D records get pellet=NaN when the rest of the batch is 3-D.
    """
    records = []
    numeric = []
    saw_legacy_shape = False
    for item in actions_history or []:
        if isinstance(item, dict):
            record = _plain(item)
            records.append(record)
            if "ecrh_W" in record and "nbi_W" in record:
                row = [float(record["ecrh_W"]), float(record["nbi_W"])]
                if "pellet_S" in record:
                    row.append(float(record["pellet_S"]))
                numeric.append(row)
            elif "ecrh_MW" in record and "nbi_MW" in record:
                row = [float(record["ecrh_MW"]) * 1e6, float(record["nbi_MW"]) * 1e6]
                if "pellet_S" in record:
                    row.append(float(record["pellet_S"]))
                numeric.append(row)
        else:
            saw_legacy_shape = True
            arr = np.asarray(item, dtype=np.float64).reshape(-1)
            if arr.size >= 2:
                numeric.append(arr[:arr.size].tolist())
                records.append({"ecrh_W": float(arr[0]), "nbi_W": float(arr[1])})
    if not numeric:
        return np.zeros((0, 2), dtype=np.float64), records
    # Pad shorter rows so all rows have the same width
    max_width = max(len(r) for r in numeric)
    padded = [r + [float('nan')] * (max_width - len(r)) for r in numeric]
    numeric_arr = np.asarray(padded, dtype=np.float64)
    if saw_legacy_shape:
        warnings.warn(
            "Legacy numeric RL action history detected. The current TORAX eval path "
            "expects dict event records with decision_t/knot_t/ecrh_W/nbi_W; "
            "numeric action arrays are still accepted for backward compatibility "
            "but should be regenerated.",
            RuntimeWarning,
            stacklevel=2,
        )
    return numeric_arr, records


def _normalizers_from_dataset(dataset_dir, replay_cache_dir=None, prefer_replay_cache=True):
    specs = describe_dataset_with_replay_cache(
        dataset_dir,
        cache_dir=replay_cache_dir,
        prefer_cache=prefer_replay_cache,
    )
    buffer = ReplayBuffer(specs["state_dim"], specs["action_dim"], specs["num_transitions"])
    load_d4rl_dataset(
        str(dataset_dir),
        buffer,
        specs["state_keys"],
        cache_dir=replay_cache_dir,
        prefer_cache=prefer_replay_cache,
    )
    return specs, normalize_buffer(buffer)


def prepare_actor_checkpoint(actor_checkpoint, output_dir, dataset_dir=None, replay_cache_dir=None, prefer_replay_cache=True):
    actor_checkpoint = Path(actor_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    ckpt = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    if "state_mean" in ckpt and "state_std" in ckpt:
        ckpt["_checkpoint_had_normalizers"] = True
        return actor_checkpoint, ckpt, None
    if dataset_dir is None:
        raise ValueError("Actor checkpoint has no state_mean/state_std. Pass --dataset_dir.")

    specs, normalizers = _normalizers_from_dataset(
        Path(dataset_dir).resolve(),
        replay_cache_dir=replay_cache_dir,
        prefer_replay_cache=prefer_replay_cache,
    )
    patched = dict(ckpt)
    patched["state_mean"] = torch.as_tensor(normalizers["state_mean"])
    patched["state_std"] = torch.as_tensor(normalizers["state_std"])
    patched.setdefault("action_max", torch.as_tensor(normalizers["action_max"]))
    patched.setdefault("state_keys", specs["state_keys"])
    patched.setdefault("state_dim", specs["state_dim"])
    patched.setdefault("action_dim", specs["action_dim"])
    patched["_checkpoint_had_normalizers"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_path = output_dir / f"{actor_checkpoint.stem}_with_eval_normalizers.pt"
    torch.save(patched, patched_path)
    return patched_path, patched, specs


def run_actor_eval_simulation(
    actor_checkpoint,
    output_dir,
    dataset_dir=None,
    project="iql-training",
    run_name=None,
    wandb_group=None,
    wandb_mode=None,
    initial_relax_state=None,
    initial_relax_cache_dir=None,
    max_loop=2,
    grid_size=51,
    device=None,
    replay_cache_dir=None,
    prefer_replay_cache=True,
    allow_cpu_jax_on_gpu=False,
    allow_mismatched_rewards=False,
    rl_segment_timeout_seconds=1800,
    rl_max_action_power_w=150.0e6,
):
    from collect_trajectories_delta import (
        build_initial_relax_cache,
        configure_tmtx,
        default_initial_relax_cache_dir,
        default_relax_geometry,
        patch_initial_relax_cache_loader,
        preflight_required_inputs,
        resolve_initial_relax_cache_path,
        resolve_seed_eqdsk_paths,
        setup_tokamaker,
        validate_jax_backend,
    )
    from OpenFUSIONToolkit.TokaMaker.pulse_design import RL_DECISION_TIMES

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_path, ckpt, dataset_specs = prepare_actor_checkpoint(
        actor_checkpoint,
        output_dir,
        dataset_dir=dataset_dir,
        replay_cache_dir=replay_cache_dir,
        prefer_replay_cache=prefer_replay_cache,
    )
    train_reward_config, train_reward_source = _load_checkpoint_reward_config(actor_checkpoint, ckpt)
    # Propagate reward_mode from the checkpoint config into the environment so that
    # default_reward_config() (and downstream compute_rewards calls) automatically
    # use the same mode the checkpoint was trained with, without requiring the caller
    # to set RL_REWARD_MODE explicitly.
    if train_reward_config and 'reward_mode' in train_reward_config:
        os.environ.setdefault('RL_REWARD_MODE', str(train_reward_config['reward_mode']))
    eval_reward_config = reward_config_to_dict(None)
    reward_config_match = _check_reward_config_match(
        actor_checkpoint=actor_checkpoint,
        train_reward_config=train_reward_config,
        train_reward_source=train_reward_source,
        eval_reward_config=eval_reward_config,
        allow_mismatched_rewards=allow_mismatched_rewards,
    )
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    cwd = Path.cwd()
    eqdsk_list = resolve_seed_eqdsk_paths(str(cwd))
    action_row = _default_action_row()
    geom = default_relax_geometry()
    relax_cache_params = None
    if initial_relax_state is None:
        legacy = None if dataset_dir is None else Path(dataset_dir) / "initial_relax_state.json"
        if legacy is not None and legacy.is_file():
            initial_relax_state = str(legacy)
        else:
            cache_dir = initial_relax_cache_dir or default_initial_relax_cache_dir()
            initial_relax_state, relax_key, relax_cache_params = resolve_initial_relax_cache_path(
                cache_dir,
                grid_size,
                geom["ne_init"],
                geom["Te_init"],
                geom["psi_sample"],
                geom["eqtimes"],
                geom["Ip_targets"],
                geom["coil_bounds"],
                geom["x_points"],
                eqdsk_list,
            )
            logger.info("Keyed initial relax cache: %s (key=%s)", initial_relax_state, relax_key)

    relax_cache_exists = initial_relax_state is not None and os.path.exists(initial_relax_state)
    preflight_required_inputs(
        str(cwd),
        eqdsk_list,
        initial_relax_cache=initial_relax_state,
        require_initial_relax_cache=relax_cache_exists,
    )
    validate_jax_backend(require_cuda_on_gpu=not allow_cpu_jax_on_gpu)

    config = {
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "eval_actor_checkpoint": str(actor_path),
        "dataset_dir": None if dataset_dir is None else str(Path(dataset_dir).resolve()),
        "output_dir": str(output_dir),
        "initial_relax_state": initial_relax_state,
        "max_loop": max_loop,
        "grid_size": grid_size,
        "checkpoint_has_normalizers": bool(ckpt.get("_checkpoint_had_normalizers", True)),
        "state_dim": int(ckpt.get("state_dim", 0) or len(ckpt["state_mean"])),
        "action_dim": int(ckpt.get("action_dim", 2)),
        "state_keys": ckpt.get("state_keys"),
        "dataset_specs": dataset_specs,
        "train_reward_config": train_reward_config,
        "train_reward_config_source": train_reward_source,
        "eval_reward_config": eval_reward_config,
        "reward_config_match": reward_config_match,
        "allow_mismatched_rewards": bool(allow_mismatched_rewards),
        "rl_segment_timeout_seconds": rl_segment_timeout_seconds,
        "rl_max_action_power_w": rl_max_action_power_w,
        "jax_compilation_cache_root": str(
            Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", Path(__file__).resolve().parent.parent / ".jax_cache")).expanduser().resolve()
        ),
        "oft_disable_jax_compile_cache": os.environ.get("OFT_DISABLE_JAX_COMPILE_CACHE", "0") == "1",
    }

    wandb_kwargs = {"project": project, "config": _plain(config), "reinit": True, "job_type": "actor_eval"}
    if run_name:
        wandb_kwargs["name"] = run_name
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    if wandb_group:
        wandb_kwargs["group"] = wandb_group
    # When mode is "disabled", skip wandb.init entirely so we don't hijack an
    # already-active training wandb session (wandb.init replaces wandb.run globally
    # even for disabled runs, breaking the caller's wandb.log calls).
    _skip_wandb = (str(wandb_kwargs.get("mode", "")).lower() == "disabled")
    if _skip_wandb:
        run = None
    else:
        run = wandb.init(**wandb_kwargs)
        wandb.define_metric("actor_eval_live/*", step_metric="actor_eval_live/decision_index")
    log_dir = output_dir / "tokamaker_torax_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _fly_t0 = [0.0]

    def log_rl_event(event):
        if event.get("event") != "decision":
            return
        try:
            idx = int(event["decision_index"])
            ecrh_mw = float(event["ecrh_MW"])
            nbi_mw = float(event["nbi_MW"])
            if not _skip_wandb:
                live_metrics = {
                    "actor_eval_live/decision_index": idx,
                    "actor_eval_live/decision_t_s": float(event["decision_t"]),
                    "actor_eval_live/ecrh_MW": ecrh_mw,
                    "actor_eval_live/nbi_MW": nbi_mw,
                    "actor_eval_live/total_heating_MW": ecrh_mw + nbi_mw,
                    "actor_eval_live/elapsed_s": time.time() - _fly_t0[0],
                    "actor_eval_live/progress": (idx + 1) / len(RL_DECISION_TIMES),
                }
                if "pellet_S" in event:
                    live_metrics["actor_eval_live/pellet_S21"] = float(event["pellet_S"]) / 1e21
                wandb.log(
                    live_metrics
                )
        except Exception as exc:
            logger.warning("Could not stream RL event to wandb: %s", exc)

    started = datetime.now().isoformat()
    try:
        mygs, _, _, _ = setup_tokamaker(str(cwd))
        if not os.path.exists(initial_relax_state):
            build_initial_relax_cache(
                mygs,
                action_row,
                initial_relax_state,
                eqdsk_list,
                geom["eqtimes"],
                geom["coil_bounds"],
                geom["x_points"],
                geom["Ip_targets"],
                geom["ne_init"],
                geom["Te_init"],
                geom["psi_sample"],
                log_dir=str(log_dir),
                grid_size=grid_size,
                params=relax_cache_params,
            )
        tmtx = configure_tmtx(
            mygs,
            action_row,
            eqdsk_list,
            geom["eqtimes"],
            geom["coil_bounds"],
            geom["x_points"],
            geom["Ip_targets"],
            geom["ne_init"],
            geom["Te_init"],
            geom["psi_sample"],
            grid_size=grid_size,
        )
        patch_initial_relax_cache_loader(tmtx)
        _fly_t0[0] = time.time()
        try:
            tmtx.fly(
                output_mode=False,
                max_loop=max_loop,
                run_name=run_name or "iql_actor_eval",
                t_ave_toggle="flattop",
                t_ave_window=25,
                relax=True,
                relax_duration=5,
                initial_relax_state=initial_relax_state,
                log_dir=str(log_dir),
                use_rl_actor=True,
                actor_checkpoint=str(actor_path),
                rl_event_callback=log_rl_event,
                rl_segment_timeout_seconds=rl_segment_timeout_seconds,
                rl_max_action_power_w=rl_max_action_power_w,
            )
        except Exception as exc:
            raise RuntimeError(
                "Actor eval failed during TORAX simulation. "
                f"{_jax_cache_workaround_note()}"
            ) from exc
        with redirect_stdout(io.StringIO()):
            summary = tmtx.summary()
        rewards = tmtx.compute_rewards()
        actions, action_records = _action_history_to_array(getattr(tmtx, "_rl_actions_history", []))
        action_dim_eval = actions.shape[1] if actions.size else 2
        default_action_max = np.ones(action_dim_eval)
        action_max = np.asarray(_plain(ckpt.get("action_max", default_action_max)), dtype=np.float64)
        if action_max.ndim == 0:
            action_max = np.full(action_dim_eval, float(action_max), dtype=np.float64)
        # Pad action_max to match actual action_dim if checkpoint predates pellet dim
        if action_max.shape[0] < action_dim_eval:
            action_max = np.concatenate([action_max, np.ones(action_dim_eval - action_max.shape[0])])
        empty_shape = (0, action_dim_eval)
        action_abs = np.abs(actions) if actions.size else np.zeros(empty_shape, dtype=np.float64)
        action_sat = action_abs >= (0.95 * action_max.reshape(1, -1)) if actions.size else np.zeros(empty_shape, dtype=bool)
        action_delta = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros(empty_shape, dtype=np.float64)
        action_delta_abs = np.abs(action_delta) if action_delta.size else action_delta
        metrics = {
            "actor_eval/reward_total": float(np.sum(rewards)),
            "actor_eval/reward_mean": float(np.mean(rewards)),
            "actor_eval/reward_min": float(np.min(rewards)),
            "actor_eval/reward_max": float(np.max(rewards)),
            "actor_eval/n_decisions": len(actions),
            "actor_eval/action_abs_mean": float(np.mean(action_abs)) if action_abs.size else 0.0,
            "actor_eval/action_delta_abs_mean": float(np.mean(action_delta_abs)) if action_delta_abs.size else 0.0,
            "actor_eval/action_saturation_rate": float(np.mean(action_sat)) if action_sat.size else 0.0,
            "actor_eval/nbi_saturation_rate": float(np.mean(action_sat[:, 1])) if action_sat.size else 0.0,
            "actor_eval/ecrh_saturation_rate": float(np.mean(action_sat[:, 0])) if action_sat.size else 0.0,
        }
        if action_dim_eval >= 3 and actions.size:
            pellet_finite = actions[:, 2][np.isfinite(actions[:, 2])]
            metrics["actor_eval/pellet_S21_mean"] = float(np.mean(pellet_finite) / 1e21) if pellet_finite.size else 0.0
            metrics["actor_eval/pellet_saturation_rate"] = float(np.mean(action_sat[:, 2])) if action_sat.size else 0.0
        for key, val in summary.items():
            if val is not None:
                metrics[f"actor_eval/{key}"] = float(val)
        if not _skip_wandb and run is not None:
            wandb.log({**metrics})
            run.summary.update(metrics)
        result = {
            "status": "success",
            "started_at": started,
            "finished_at": datetime.now().isoformat(),
            "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
            "eval_actor_checkpoint": str(actor_path),
            "initial_relax_state": _bundle_safe(initial_relax_state),
            "summary": _plain(summary),
            "reward_total": float(np.sum(rewards)),
            "rewards": _plain(rewards),
            "actions": _plain(actions),
            "action_records": action_records,
            "metrics": _plain(metrics),
            "train_reward_config": _plain(train_reward_config),
            "train_reward_config_source": train_reward_source,
            "eval_reward_config": _plain(eval_reward_config),
            "reward_config_match": reward_config_match,
            "output_dir": str(output_dir),
    }
        bundle = {
            "state": _bundle_safe(tmtx._state),
            "tm_times": list(getattr(tmtx, "_tm_times", [])),
            "current_loop": getattr(tmtx, "_current_loop", None),
            "flattop": _bundle_safe(getattr(tmtx, "_flattop", None)),
            "coil_bounds": _bundle_safe(getattr(tmtx, "_coil_bounds", None)),
            "results": _bundle_safe(getattr(tmtx, "_results", None)),
            "output_mode": getattr(tmtx, "_output_mode", None),
            "actor_checkpoint": str(actor_path),
            "initial_relax_state": initial_relax_state,
            "grid_size": grid_size,
            "max_loop": max_loop,
        }
        bundle_path = output_dir / "actor_eval_bundle.pkl"
        with bundle_path.open("wb") as f:
            pickle.dump(bundle, f)
        result["bundle_path"] = str(bundle_path)

        reward_config_path = output_dir / "actor_eval_reward_config.json"
        reward_config_payload = {
            "eval_reward_config": _plain(eval_reward_config),
            "train_reward_config": _plain(train_reward_config),
            "train_reward_config_source": train_reward_source,
            "reward_config_match": bool(reward_config_match),
            "allow_mismatched_rewards": bool(allow_mismatched_rewards),
        }
        with reward_config_path.open("w") as f:
            json.dump(reward_config_payload, f, indent=2, sort_keys=True)
        if not _skip_wandb: wandb.save(str(reward_config_path))
        result["reward_config_path"] = str(reward_config_path)

        result_path = output_dir / "actor_eval_summary.json"
        with result_path.open("w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        if not _skip_wandb: wandb.save(str(result_path))
        actions_path = output_dir / "actor_eval_actions.json"
        with actions_path.open("w") as f:
            json.dump(
                {
                    "actions": _plain(actions),
                    "action_records": action_records,
                    "action_max": _plain(action_max),
                    "metrics": _plain(metrics),
                },
                f,
                indent=2,
                sort_keys=True,
            )
        if not _skip_wandb: wandb.save(str(actions_path))
        result["tmtx"] = tmtx
        return result
    finally:
        if not _skip_wandb:
            wandb.finish()


def _default_action_row():
    from collect_trajectories_delta import DECISION_TIMES
    from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX

    actions = []
    for decision_t in DECISION_TIMES:
        knot_t = decision_t + 20
        ecrh_w, nbi_w = TokaMaker_TORAX._rl_default_action_w_at_time(knot_t)
        actions.append([ecrh_w, nbi_w])
    return np.asarray(actions, dtype=np.float64)
