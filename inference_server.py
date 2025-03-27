import socket
import torch
import numpy as np
import logging
from vanet_env import VANETCommEnv
from sac_agent import SACAgent

logging.basicConfig(level=logging.INFO)

env = VANETCommEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)
agent.policy_net.load_state_dict(torch.load("sac_policy_vanet.pth"))
agent.policy_net.eval()
logging.info("Model loaded and inference ready.")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

while True:
    data = conn.recv(1024)
    if not data:
        break
    input_array = np.frombuffer(data, dtype=np.float32)  # [beaconRate, txPower, CBR, SNR, MCS]
    beaconRate, txPower, cbr, snr, mcs = input_array
    state = np.array([beaconRate, txPower, cbr, snr], dtype=np.float32)
    action = agent.select_action(state)
    scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (
        env.action_space.high - env.action_space.low)
    scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)
    new_beacon_rate = scaled_action[0]
    new_tx_power = scaled_action[1]
    output_array = np.array([new_beacon_rate, new_tx_power, mcs], dtype=np.float32)
    conn.sendall(output_array.tobytes())
    logging.info(f"Responded with: Beacon {new_beacon_rate:.2f}, TxPower {new_tx_power:.2f}, MCS {mcs}")

conn.close()
server.close()
logging.info("Connection closed.")
