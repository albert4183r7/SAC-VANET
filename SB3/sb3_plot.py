import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_latest_tb_log(log_root):
    for root, dirs, files in os.walk(log_root):
        for file in files:
            if file.startswith("events.out.tfevents"):
                return root
    return None

def load_tensorboard_log(log_dir, tag='rollout/ep_rew_mean'):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({'Step': steps, 'Reward': values})

def load_intermediate_eval(path="SB3/logs/eval/evaluations.npz"):
    if os.path.exists(path):
        data = np.load(path)
        return pd.DataFrame({
            'timesteps': data['timesteps'].flatten(),
            'mean_reward': data['results'].mean(axis=1),
            'std_reward': data['results'].std(axis=1)
        })
    return None

def plot_training_and_evaluation(train_df, eval_df=None, title="SB3: Training & Intermediate Evaluation"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_df['Step'], train_df['Reward'], label='Training Reward', color='blue')

    if eval_df is not None:
        plt.plot(eval_df['timesteps'], eval_df['mean_reward'], 'o-', label='Eval Reward (During Training)', color='orange')
        plt.fill_between(eval_df['timesteps'],
                         eval_df['mean_reward'] - eval_df['std_reward'],
                         eval_df['mean_reward'] + eval_df['std_reward'],
                         alpha=0.2, color='orange')

    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run plotting
log_root = "SB3/logs/"
log_path = find_latest_tb_log(log_root)
train_df = load_tensorboard_log(log_path) if log_path else None
eval_df = load_intermediate_eval()

if train_df is not None:
    plot_training_and_evaluation(train_df, eval_df)
else:
    print("Training logs not found.")
