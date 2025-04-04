import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_sb3_log(log_root, tag='rollout/ep_rew_mean'):
    for root, _, files in os.walk(log_root):
        for file in files:
            if file.startswith("events.out.tfevents"):
                ea = EventAccumulator(root)
                ea.Reload()
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                return pd.DataFrame({'Step': steps, 'Reward': values})
    return None

def load_custom_log(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def plot_comparison(sb3_df, custom_df, title="Reward Comparison: SB3 SAC vs Custom SAC"):
    plt.figure(figsize=(10, 6))

    if sb3_df is not None:
        plt.plot(sb3_df['Step'], sb3_df['Reward'], label='SB3 SAC', color='blue')

    if custom_df is not None:
        plt.plot(custom_df['timesteps'], custom_df['ep_rew_mean'], label='Custom SAC', color='green')

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sb3_log_path = "SB3/logs"
    custom_log_path = "custom/logs/training_rewards.csv"

    sb3_df = load_sb3_log(sb3_log_path)
    custom_df = load_custom_log(custom_log_path)

    if sb3_df is not None or custom_df is not None:
        plot_comparison(sb3_df, custom_df)
    else:
        print("No logs found for SB3 or Custom.")