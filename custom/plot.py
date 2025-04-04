import os
import pandas as pd
import matplotlib.pyplot as plt

def load_training_log(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_intermediate_eval_log(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def plot_custom_logs(train_df, intermediate_df=None, title="SAC Training & Intermediate Evaluation"):
    plt.figure(figsize=(10, 6))

    if train_df is not None:
        plt.plot(train_df['timesteps'], train_df['ep_rew_mean'], label='Training Reward', color='blue')

    if intermediate_df is not None:
        intermediate_df['timesteps'] = intermediate_df['episode'] * 100
        plt.plot(intermediate_df['timesteps'], intermediate_df['mean_eval_reward'],
                 label='Intermediate Eval Reward', color='orange', marker='x')

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
