import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VANETCommEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([1.0, 0.0, 0.1, 0.0]),  # [beaconRate, txPower, CBR, SNR]
            high=np.array([10.0, 20.0, 1.0, 30.0]),
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
            np.random.uniform(1.0, 10.0),    # beaconRate
            np.random.uniform(0.0, 20.0),    # txPower
            np.random.uniform(0.1, 1.0),     # CBR
            np.random.uniform(0.0, 30.0)     # SNR
        ], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        beacon_rate, tx_power = action
        prev_tx_power = self.state[1]
        self.step_count += 1

        # Aproksimasi CBR dan SNR dari beacon_rate dan tx_power
        cbr = self.state[2]
        snr = self.state[3]

        # --- Hitung reward ---
        omega_c = 2
        omega_p = 0.25
        target_cbr = 0.6
        target_snr = 27.5
        snr_tolerance = 5

        # CBR reward
        g_cbr = -np.sign(cbr - target_cbr) * cbr
        reward_cbr = omega_c * g_cbr

        # Delta Power reward
        reward_power = -omega_p * abs(tx_power - prev_tx_power)

        # SNR reward (continuous)
        snr_deviation = abs(snr - target_snr)
        if snr_deviation <= snr_tolerance:
            reward_snr = 10 * (1 - snr_deviation / snr_tolerance)
        else:
            reward_snr = -5 * (snr_deviation - snr_tolerance)

        # Bonus kombinasi ideal
        if 0.55 <= cbr <= 0.65 and 25 <= snr <= 30:
            reward_bonus = 5
        else:
            reward_bonus = 0

        reward = reward_cbr + reward_power + reward_snr + reward_bonus

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
