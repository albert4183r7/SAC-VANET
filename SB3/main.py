import argparse
import subprocess
from SB3.sb3_train_and_evaluate import train_sb3_vanet
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_eval_log_npz(npz_path):
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]  # shape: (n_eval, n_runs)
        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)
        return pd.DataFrame({
            "timesteps": timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        })
    return None


def find_latest_tb_log(log_root):
    for root, dirs, files in os.walk(log_root):
        for file in files:
            if file.startswith("events.out.tfevents"):
                return root
    return None

def load_tensorboard_log(log_dir, tag='rollout/ep_rew_mean'):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({'Step': steps, 'Reward': values})

def plot_training_and_evaluation(train_df, eval_df=None, title="SB3: Training & Eval Reward"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['Step'], train_df['Reward'], label='Training Reward (SB3)', color='blue')

    if eval_df is not None:
        plt.plot(eval_df['timesteps'], eval_df['mean_reward'], 'o-', label='Eval Reward (During Training)', color='orange')
        if 'std_reward' in eval_df.columns:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC-VANET SB3 Agent CLI")
    parser.add_argument('--train', action='store_true', help='Train the SB3 SAC agent')
    parser.add_argument('--plot', action='store_true', help='Plot training and evaluation rewards')
    parser.add_argument('--serve', action='store_true', help='Run inference server')

    args = parser.parse_args()

    if args.train:
        train_sb3_vanet()

    if args.plot:
        log_path = find_latest_tb_log("SB3/logs")
        train_df = load_tensorboard_log(log_path) if log_path else None
        eval_df = load_eval_log_npz("SB3/logs/eval/evaluations.npz")  # this is SB3's default

        if eval_df is None and os.path.exists("SB3/logs/eval/evaluations.npz"):
            import numpy as np
            data = np.load("SB3/logs/eval/evaluations.npz")
            eval_df = pd.DataFrame({
                'timesteps': data['timesteps'].flatten(),
                'mean_reward': data['results'].mean(axis=1),
                'std_reward': data['results'].std(axis=1)
            })

        if train_df is not None:
            plot_training_and_evaluation(train_df, eval_df)
        else:
            print("Training log not found.")

    if args.serve:
        subprocess.run([sys.executable, "sb3_infer_server.py"])
