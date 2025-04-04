import torch
import logging
from vanet_env import VANETCommEnv
from custom.sac_agent import SACAgent
import numpy as np
import csv
import os

logging.basicConfig(level=logging.INFO)

def train_sac_on_environment():
    env = VANETCommEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    logging.info("Training started.")

    num_episodes = 200
    explore_episodes = 10
    eval_interval = 20

    log_path = "custom/logs/training_rewards.csv"
    eval_log_path = "custom/logs/intermediate_eval.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as f_train, open(eval_log_path, "w", newline="") as f_eval:
        train_writer = csv.writer(f_train)
        eval_writer = csv.writer(f_eval)
        train_writer.writerow(["episode", "timesteps", "ep_rew_mean"])
        eval_writer.writerow(["episode", "mean_eval_reward"])

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
                state = next_state
                total_reward += reward
                if done:
                    break
            logging.info(f"Episode {ep+1} finished. Total reward: {total_reward:.2f}")
            train_writer.writerow([ep + 1, (ep + 1) * env.max_steps, total_reward])

            # Evaluasi berkala
            if (ep + 1) % eval_interval == 0:
                eval_rewards = []
                for _ in range(5):
                    state, _ = env.reset()
                    total_eval_reward = 0
                    for _ in range(env.max_steps):
                        action = agent.select_action(state)
                        scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (env.action_space.high - env.action_space.low)
                        scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)
                        next_state, reward, terminated, truncated, _ = env.step(scaled_action)
                        total_eval_reward += reward
                        state = next_state
                        if terminated or truncated:
                            break
                    eval_rewards.append(total_eval_reward)
                mean_eval_reward = np.mean(eval_rewards)
                logging.info(f"[Evaluation @Episode {ep+1}] Mean Eval Reward: {mean_eval_reward:.2f}")
                eval_writer.writerow([ep + 1, mean_eval_reward])

    torch.save(agent.policy_net.state_dict(), "model/custom.pth")
    logging.info("Model saved to model/custom.pth")
    return agent
