import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VANETCommEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([1.0, 0.0, 0.0, 0.0]),  # [beaconRate, txPower, CBR, SNR]
            high=np.array([10.0, 20.0, 1.0, 50.0]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([1.0, 0.0]),  # [beaconRate, txPower]
            high=np.array([10.0, 20.0]),
            dtype=np.float32
        )
        self.state = None
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            np.random.uniform(1.0, 10.0),     # beaconRate
            np.random.uniform(0.0, 20.0),     # txPower
            np.random.uniform(0.1, 0.4),      # CBR
            np.random.uniform(25, 35)         # SNR
        ], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        beacon_rate, tx_power = action
        self.step_count += 1

        cbr = min(1.0, self.state[2] + (beacon_rate * tx_power * 0.0005))
        snr = self.state[3] + (tx_power * 0.1) - (self.state[0] * 0.05)

        reward = 0
        if cbr > 0.6:
            reward -= (cbr - 0.6) * 10
        if snr < 25:
            reward -= (25 - snr) * 5
        if 0.3 < cbr <= 0.6 and snr >= 25:
            reward += 10

        next_state = np.array([
            beacon_rate,
            tx_power,
            cbr,
            snr
        ], dtype=np.float32)

        done = self.step_count >= self.max_steps
        self.state = next_state
        return next_state, reward, done, False, {}

    def render(self):
        print(f"Step {self.step_count} | State: {self.state}")
