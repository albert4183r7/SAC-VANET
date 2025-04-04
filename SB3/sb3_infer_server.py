import socket
import json
import numpy as np
from stable_baselines3 import SAC
from vanet_env import VANETCommEnv

model = SAC.load("model/sac_sb3_vanet")
env = VANETCommEnv()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
print("[SB3 Server] Listening on port 5000...")
conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    data = conn.recv(1024)
    if not data:
        break
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
    except Exception as e:
        print(f"Error: {e}")
        continue

conn.close()
server.close()
print("Server closed.")