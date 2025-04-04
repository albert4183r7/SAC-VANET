import socket
import torch
import numpy as np
import logging
import json
from vanet_env import VANETCommEnv
from sac_agent import SACAgent

logging.basicConfig(level=logging.INFO)

env = VANETCommEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)
agent.policy_net.load_state_dict(torch.load("model/custom.pth"))
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
        logging.info("No data received, closing connection.")
        break
    logging.info(f"Received data: {data.decode()}")
    
    # input_array = np.frombuffer(data, dtype=np.float32)  # [beaconRate, txPower, CBR, SNR, MCS]
    # beaconRate, txPower, cbr, snr, mcs = input_array
    
    rlData = json.loads(data.decode('utf-8'))

    txPower = float(rlData['transmissionPower'])
    beaconRate = float(rlData['beaconRate'])
    cbr = float(rlData['CBR'])
    snr = float(rlData['SNR'])
    mcs = float(rlData['MCS'])
    
    state = np.array([beaconRate, txPower, cbr, snr], dtype=np.float32)
    action = agent.select_action(state)
    
    scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (
        env.action_space.high - env.action_space.low)
    scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)
    
    new_beacon_rate = scaled_action[0]
    new_tx_power = scaled_action[1]
    
    # output_array = np.array([new_beacon_rate, new_tx_power, mcs], dtype=np.float32)
    # conn.sendall(output_array.tobytes())
    
    response = json.dumps({
        "transmissionPower": float(new_tx_power),
        "beaconRate": float(new_beacon_rate),
        "MCS": mcs
    }).encode('utf-8')
    conn.sendall(response)

    logging.info(f"Sent optimized: txPower={new_tx_power:.2f}, beaconRate={new_beacon_rate:.2f}, MCS={mcs}")

conn.close()
server.close()
logging.info("Connection closed.")
