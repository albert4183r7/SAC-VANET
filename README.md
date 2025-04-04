# SAC-VANET: Soft Actor-Critic for VANET Communication Optimization

This project applies **Soft Actor-Critic (SAC)**, a reinforcement learning algorithm, to optimize **Beacon Rate** and **Transmit Power** in **Vehicular Ad-hoc Networks (VANETs)**. The goal is to learn optimal transmission strategies by adjusting **Beacon Rate** and **Transmit Power**, in order to maximize communication quality and efficiency in VANETs, as reflected by metrics like **Channel Busy Ratio (CBR)** and **Signal-to-Noise Ratio (SNR)**.

There are two types of SAC implementations:
- **Custom**: Fully custom SAC agent built from scratch.
- **SB3**: SAC agent built using [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3).

---

## 📁 Project Structure

```
SAC-VANET/
│
├── custom/                  ← Custom SAC agent
│   ├── inference_server.py       # Inference server using custom agent
│   ├── main.py                   # Entry point for training/plot/inference
│   ├── sac_agent.py              # SAC agent, policy, Q-networks, replay buffer
│   ├── train_and_evaluate.py     # Training and evaluation logic
│   ├── plot.py                   # Visualize reward from custom logs
│   └── logs/                     # Logs for custom agent
│       ├── training_rewards.csv
│       └── eval_results.csv
│
├── SB3/                    ← SB3-based SAC agent
│   ├── main.py                     # Entry point for SB3 training/plot/inference
│   ├── sb3_infer_server.py         # Inference server using SB3 agent
│   ├── sb3_train_and_evaluate.py   # SB3 model training and evaluation logic
│   ├── sb3_plot.py                 # Visualize reward from TensorBoard logs
│   ├── logs/                       # TensorBoard logs for SB3
│       ├── eval/
│       └── SAC_VANET_1/
│
├── vanet_env.py            # custom VANET environment
├── model/                  # Folder for saved models (created during training)
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── .gitignore
```

---

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Dependencies:
```
torch>=2.0.0
numpy>=1.23
gymnasium>=0.29.1
stable-baselines3>=2.2.1
tensorboard
matplotlib
pandas
```

---

## 🚀 How to Use

**Train the model:**
```bash
python -m {custom/SB3}.main --train
```

**Plot Training Result (Reward):**
```bash
python -m {custom/SB3}.main --plot
```

**Run the inference server:**
```bash
python -m {custom/SB3}.main --serve
```

---

## 📄 File Details

- `vanet_env.py`: Defines custom VANET gym environment.

### `custom/`
- `main.py`: CLI entry point for training, evaluation, and plotting.
- `sac_agent.py`: Implements the SAC agent from scratch (policy, critic networks, buffer).
- `train_and_evaluate.py`: Training and evaluation functions for the custom SAC agent with CSV logging.
- `inference_server.py`: Socket server to handle live state input and return optimized output using trained custom model.
- `plot.py`: Visualizes training and evaluation reward curves for the custom agent.


### `SB3/`
- `main.py`: CLI entry point for training, evaluation, and plotting using SB3.
- `sb3_train_and_evaluate.py`: Training logic using Stable-Baselines3 SAC implementation.
- `sb3_infer_server.py`: Inference server using trained SB3 model.
- `sb3_plot.py`: Visualizes training and evaluation reward curves for the SB3 SAC agent.

---

## ✨ Logs

- `custom/logs/`: stores `training_rewards.csv` and `eval_results.csv`
- `SB3/logs/`: TensorBoard logs, visualizable with `tensorboard` or via plot script

---

## 🧠 Notes

- Make sure you run from the project root to ensure relative imports and log paths work.
- For SB3 logging, use TensorBoard:
```bash
tensorboard --logdir SB3/logs/
```