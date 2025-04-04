import argparse
from custom.train_and_evaluate import train_sac_on_environment
from custom.plot import plot_custom_logs
import pandas as pd
import os
import subprocess
import sys

def load_training_log(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_intermediate_eval_log(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC-VANET Custom Agent CLI")
    parser.add_argument('--train', action='store_true', help='Train the custom SAC agent')
    parser.add_argument('--plot', action='store_true', help='Plot training and intermediate evaluation rewards')
    parser.add_argument("--serve", action="store_true", help="Run inference socket server")

    args = parser.parse_args()

    if args.train:
        train_sac_on_environment()

    if args.plot:
        train_df = load_training_log("custom/logs/training_rewards.csv")
        intermediate_df = load_intermediate_eval_log("custom/logs/intermediate_eval.csv")

        if train_df is not None:
            plot_custom_logs(train_df, intermediate_df)
        else:
            print("Training log not found.")

    if args.serve:
        subprocess.run([sys.executable, "inference_server.py"])
