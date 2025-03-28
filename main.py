import argparse
from train_and_evaluate import train_sac_on_environment, evaluate_sac_model
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the SAC model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the SAC model")
    parser.add_argument("--serve", action="store_true", help="Run inference socket server")
    args = parser.parse_args()

    if args.train:
        train_sac_on_environment()
    if args.eval:
        evaluate_sac_model()
    if args.serve:
        subprocess.run([sys.executable, "inference_server.py"])
