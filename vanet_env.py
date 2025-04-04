import gymnasium as gym
from gymnasium import spaces
import numpy as np

class VANETCommEnv(gym.Env):
    def __init__(self):
        super(VANETCommEnv, self).__init__()

        self.action_space = spaces.Box(low=np.array([1.0, 1.0]), high=np.array([10.0, 10.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([1.0, 1.0, 0.1, 0.0]),
                                            high=np.array([10.0, 10.0, 1.0, 30.0]), dtype=np.float32)

        self.max_steps = 100
        self.target_cbr = 0.5
        self.target_snr = 15.0
        self.snr_tolerance = 5.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([
            np.random.uniform(1.0, 10.0),   # beacon_rate
            np.random.uniform(1.0, 10.0),   # tx_power
            0.5,                            # initial cbr
            15.0                            # initial snr
        ], dtype=np.float32)
        self.prev_tx_power = self.state[1]
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        beacon_rate = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        tx_power = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        # Simulate environment response
        cbr = np.clip(0.1 + 0.08 * beacon_rate + np.random.normal(0, 0.02), 0.1, 1.0)
        snr = 15.0 \
            - 0.8 * (beacon_rate - 5.0)**2 \
            - 0.8 * (tx_power - 5.0)**2 \
            + np.random.normal(0, 1.0)
        snr = np.clip(snr, 0.0, 30.0)
        
        # --- Balanced Reward Function ---
        # CBR reward
        reward_cbr = 10 * (1 - abs(cbr - self.target_cbr) / self.target_cbr)

        # SNR reward
        snr_deviation = abs(snr - self.target_snr)
        if snr_deviation <= self.snr_tolerance:
            reward_snr = 10 * (1 - snr_deviation / self.snr_tolerance)
        else:
            reward_snr = -2 * (snr_deviation - self.snr_tolerance)

        reward_snr = max(reward_snr, -15)

        # Power stability reward
        reward_power = -0.2 * abs(tx_power - self.prev_tx_power)

        # Bonus
        reward_bonus = 0
        if abs(cbr - self.target_cbr) < 0.05 and abs(snr - self.target_snr) < 2.0:
            reward_bonus = 5
        
        reward = reward_cbr + reward_snr + reward_power + reward_bonus
        
        # print("bc", beacon_rate)
        # print("tx",tx_power)
        # print("cbr", cbr)
        # print("snr", snr)
        # print("reward_cbr", reward_cbr)
        # print("reward_snr", reward_snr)
        
        next_state = np.array([
            beacon_rate,
            tx_power,
            cbr,
            snr
        ], dtype=np.float32)

        self.prev_tx_power = tx_power
        self.state = next_state
        terminated = self.current_step >= self.max_steps
        truncated = False

        return next_state, reward, terminated, truncated, {}