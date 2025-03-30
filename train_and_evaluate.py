import torch
import logging
from vanet_env import VANETCommEnv
from sac_agent import SACAgent
import numpy as np

logging.basicConfig(level=logging.INFO)

def train_sac_on_environment():
    env = VANETCommEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    logging.info("Training started.")

    num_episodes = 500
    explore_episodes = 10  # <- episode awal pakai random action
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(env.max_steps):
            if ep < explore_episodes:
                action = np.random.uniform(low=-1, high=1, size=action_dim)
            else:
                action = agent.select_action(state)
            
            scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (env.action_space.high - env.action_space.low)
            scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            agent.store_transition(state, scaled_action, reward, next_state, done)
            agent.train()
            print(f"Step {t+1}: Beacon Rate = {scaled_action[0]:.2f}, Tx Power = {scaled_action[1]:.2f}")
            state = next_state
            total_reward += reward
            if done:
                break
        logging.info(f"Episode {ep+1} finished. Total reward: {total_reward:.2f}")

    torch.save(agent.policy_net.state_dict(), "sac_policy_vanet#1.pth")
    logging.info("Model saved to sac_policy_vanet.pth")
    return agent


def evaluate_sac_model(model_path="sac_policy_vanet.pth", num_episodes=100):
    env = VANETCommEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    logging.info("Evaluation started.")

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        print(f"\nEvaluasi Episode {ep+1}")
        for t in range(env.max_steps):
            action = agent.select_action(state)
            scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (
                env.action_space.high - env.action_space.low)
            scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            print(f"Step {t+1}: Beacon Rate = {scaled_action[0]:.2f}, Tx Power = {scaled_action[1]:.2f}")
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        logging.info(f"Eval Episode {ep+1} Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    train_sac_on_environment()
    evaluate_sac_model()
