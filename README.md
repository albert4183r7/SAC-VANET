# SAC-VANET: Soft Actor-Critic for VANET Communication Optimization

This project uses **Soft Actor-Critic (SAC)**, a reinforcement learning algorithm, to optimize the **Beacon Rate** and **Transmit Power** in a **VANET (Vehicular Ad-hoc Network)** environment. The system is designed to ensure that **Channel Busy Ratio (CBR)** and **Signal-to-Noise Ratio (SNR)** stay within acceptable limits.

## Project Structure

- **`vanet_env.py`**: Defines the VANET communication environment for training and inference.
- **`sac_agent.py`**: Contains the SAC agent, including policy and Q-networks, along with the replay buffer.
- **`train_and_evaluate.py`**: Includes functions for training the SAC model and evaluating its performance.
- **`inference_server.py`**: A socket server to run the trained model and receive real-time data for prediction.
- **`main.py`**: Entry point for running training, evaluation, or inference through command-line arguments.
- **`sac_policy_vanet.pth`**: The saved model from training (will be generated during training).

## Requirements

This project requires the following Python packages:

```bash
torch>=2.0.0
numpy>=1.23
gymnasium>=0.29.1
```

To install all dependencies, use:
```bash
pip install -r requirements.txt
```

## How to Use
### 1. Training the Model
To train the SAC model, run the following command:
```bash
python main.py --train
```

This will:
- Start training for 10 episodes.
- Save the trained model to sac_policy_vanet.pth.


### 2. Evaluating the Model

To evaluate the trained model, run:
```bash
python main.py --eval
```

This will:
- Evaluate the model over 3 episodes and print the results (Beacon Rate, Transmit Power, etc.).


### 3. Running the Inference Server

To run the inference server, which listens for incoming data on localhost:5000, use:
```
python main.py --serve
```

The server will:
- Accept incoming data.
- Process the data with the trained model.
- Return the optimized values.


## File Details
### vanet_env.py
<p>Defines the custom environment for VANET communication. It uses gymnasium for easy integration with reinforcement learning algorithms. The environment simulates a communication scenario where the agent optimizes the Beacon Rate and Transmit Power while keeping CBR and SNR in check. </p>

### sac_agent.py
<p>Contains the definition of the SAC agent:
- Policy Network: For selecting actions.
- Q-Networks: To estimate the Q-value of actions.
- Replay Buffer: To store past experiences during training.</p>

### train_and_evaluate.py
<p>Contains functions for training the SAC agent and evaluating its performance. The trained model is saved as sac_policy_vanet.pth for later use in inference.</p>

### inference_server.py
<p>This script starts a server listening on localhost:5000, receives state observations, and sends back the optimized actions (beacon rate, transmit power, and MCS).</p>

### main.py
<p>This is the entry point for running training, evaluation, or inference. You can control which action to run through command-line arguments:</p>
<div>--train: Train the model.</div>
<div>--eval: Evaluate the model.</div>
<div>--serve: Run the inference server.</div>
