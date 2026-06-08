
We have IQL in IQL.py. Can we try new algorithms?

- Go off our existing collected trajectories, do not collect new ones because they take forever:
	`run_prev_action_wide_ecrh_20260602_000037/`
	`run_prev_action_full_20260601_192826/`
	Or others, idk which ones the best

- You can start off as many RL runs as you want because they're short. For GPU runs, use `jag-standard,jag-hi` to get priority as your partition.
- You can change the reward params and then use `update_trajectories.py`. Remember, what we're looking for is good performance on evals which run the simulator.
- For CPU runs, make sure `sc-loprio` is part of the partition used.
	- Note that these jobs can be pre-empted and restarted.

- Make notes in ./docs as needed.
- Record all results stats, even for badly performing runs, in RESULTS.md
- Have runs save to wandb with descriptive names indicating the things you changed and the type of algorithm.
