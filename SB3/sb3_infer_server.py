import socket
import json
import logging
import numpy as np
from stable_baselines3 import SAC
from vanet_env import VANETCommEnv

logging.basicConfig(level=logging.INFO)

model = SAC.load("model/sac_sb3_vanet")
logging.info("Model loaded and inference ready.")
env = VANETCommEnv()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("[Server] Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

while True:
    data = conn.recv(1024)
    if not data:
        logging.info("No data received, closing connection.")
        break
    logging.info(f"Received data: {data.decode()}")
    try:
        rlData = json.loads(data.decode('utf-8'))
        txPower = float(rlData['transmissionPower'])
        beaconRate = float(rlData['beaconRate'])
        cbr = float(rlData['CBR'])
        snr = float(rlData['SNR'])
        mcs = float(rlData['MCS'])

        obs = np.array([beaconRate, txPower, cbr, snr], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        response = json.dumps({
            "transmissionPower": float(action[1]),
            "beaconRate": float(action[0]),
            "MCS": mcs
        }).encode('utf-8')
        conn.sendall(response)
        
        logging.info(f"Sent optimized: txPower={action[1]:.2f}, beaconRate={action[0]:.2f}, MCS={mcs}")
    
    except Exception as e:
        logging.info(f"Error: {e}")
        continue

conn.close()
server.close()
logging.info("Server closed.")