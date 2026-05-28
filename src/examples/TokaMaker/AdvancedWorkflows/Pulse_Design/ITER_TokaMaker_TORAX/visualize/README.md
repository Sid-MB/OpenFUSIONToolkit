# Trajectory Visualization

Generate static plots from per-trajectory datasets produced by
`collect_trajectories_delta.py`. The script supports both current Zarr stores
and older flat `trajectory_*.json` datasets.

```bash
uv run python visualize/plot_trajectories.py \
  rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr_2000_20260527_123853 \
  --trajectory-ids 0 338 1330 \
  --skip-profiles
```

Without a dataset argument, the script uses the newest
`rl_dataset_delta_sampling_maxloop=2_grid_51_full_zarr*` dataset with completed
Zarr stores. Outputs are written to `visualize/out/<dataset-name>/`.

The main outputs are:

- `summary_distributions.png`: histograms of Q, flux, and safety summaries.
- `summary_scatter.png`: Q/flux and safety scatter plots across trajectories.
- `sample_actions_rewards.png`: sampled action schedules and reward traces.
- `sample_safety_traces.png`: q95, beta_N, and Greenwald-fraction traces.
- `trajectory_<id>_scalars.png`: full TORAX scalar traces for selected runs.
- `trajectory_<id>_profiles.png`: profile heatmaps for selected runs unless
  `--skip-profiles` is set. Profile heatmaps require Zarr stores and are skipped
  for JSON-only datasets.
