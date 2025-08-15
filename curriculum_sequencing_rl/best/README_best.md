# Running Best Hyperparameters (Test-Set Evaluation)

This snippet shows how to:
- Launch any single model using the curated best hyperparameters from `best_hyperparams.json`.
- Run the orchestrator directly with explicit flags per model.
- Plot a bar chart of best accuracies.

All results append to: `curriculum_sequencing_rl/experiment_metrics.csv`.

## Option A: Use the launcher (recommended)

Run any model with its best saved params and seed:

PowerShell (from repo root):

```powershell
python .\curriculum_sequencing_rl\best\run_best.py --model ql --eval_episodes 1000
python .\curriculum_sequencing_rl\best\run_best.py --model dqn --eval_episodes 1000
python .\curriculum_sequencing_rl\best\run_best.py --model a2c --eval_episodes 1000
python .\curriculum_sequencing_rl\best\run_best.py --model a3c --eval_episodes 1000
python .\curriculum_sequencing_rl\best\run_best.py --model ppo --eval_episodes 1000
```

Notes:
- Defaults: data=`preprocessed_kt_data.csv`, metrics CSV=`curriculum_sequencing_rl/experiment_metrics.csv`, include chance baseline.
- Override seed if desired: add `--seed 42`.
- Preview params without running: add `--dry_run`.

## Option B: Call the orchestrator directly with explicit flags

These reproduce the best runs (test evaluation). All use `--reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --no_chance`.

Q-Learning (acc≈0.708, seed=123):
```powershell
python .\curriculum_sequencing_rl\curriculum_rl_experiments.py --models ql --ql_epochs 120 --ql_alpha 0.7 --ql_gamma 0.995 --ql_eps_start 0.6 --ql_eps_end 0.0 --ql_eps_decay_epochs 20 --ql_val_episodes 300 --reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --metrics_csv ".\curriculum_sequencing_rl\experiment_metrics.csv" --seed 123 --no_chance
```

DQN (acc≈0.928, seed=123):
```powershell
python .\curriculum_sequencing_rl\curriculum_rl_experiments.py --models dqn --dqn_episodes 400 --dqn_lr 2e-4 --dqn_gamma 0.997 --dqn_batch_size 512 --dqn_buffer_size 120000 --dqn_hidden_dim 512 --dqn_eps_start 1.0 --dqn_eps_end 0.01 --dqn_eps_decay_steps 20000 --dqn_target_tau 0.01 --dqn_target_update_interval 1 --dqn_val_episodes 300 --reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --metrics_csv ".\curriculum_sequencing_rl\experiment_metrics.csv" --seed 123 --no_chance
```

A2C (acc≈0.656, seed=123, heavy BC):
```powershell
python .\curriculum_sequencing_rl\curriculum_rl_experiments.py --models a2c --a2c_episodes 400 --a2c_lr 3e-4 --a2c_entropy 0.02 --a2c_value_coef 0.5 --a2c_bc_warmup 8 --a2c_bc_weight 2.0 --a2c_batch_episodes 8 --reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --metrics_csv ".\curriculum_sequencing_rl\experiment_metrics.csv" --seed 123 --no_chance
```

A3C (acc≈0.654, seed=123):
```powershell
python .\curriculum_sequencing_rl\curriculum_rl_experiments.py --models a3c --a3c_episodes 400 --a3c_lr 3e-4 --a3c_entropy 0.02 --a3c_value_coef 0.5 --a3c_gae_lambda 0.95 --a3c_bc_warmup 8 --a3c_bc_weight 2.0 --a3c_rollouts 8 --reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --metrics_csv ".\curriculum_sequencing_rl\experiment_metrics.csv" --seed 123 --no_chance
```

PPO (acc≈0.779, seed=456):
```powershell
python .\curriculum_sequencing_rl\curriculum_rl_experiments.py --models ppo --ppo_episodes 400 --ppo_lr 3e-4 --ppo_epochs 8 --ppo_batch_episodes 16 --ppo_minibatch_size 1024 --ppo_entropy 0.02 --ppo_value_coef 0.5 --ppo_gae_lambda 0.95 --ppo_bc_warmup 8 --ppo_bc_weight 2.0 --reward_correct_w 1.0 --reward_score_w 0.0 --eval_episodes 1000 --metrics_csv ".\curriculum_sequencing_rl\experiment_metrics.csv" --seed 456 --no_chance
```

## Plot best accuracies

Generate a bar chart from `best_hyperparams.json`:
```powershell
python .\curriculum_sequencing_rl\best\plot_best_accuracies.py --output .\curriculum_sequencing_rl\best\best_accuracies.png
```
The figure is saved to `curriculum_sequencing_rl/best/best_accuracies.png`.
