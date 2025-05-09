import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
import ale_py
from gymnasium.wrappers.frame_stack import FrameStack
from game import Network, Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_dcqn():
    try:
        env = gym.make('ALE/MsPacman-v5', full_action_space=False, render_mode=None)
        env = FrameStack(env, 4)
        logging.info("Successfully initialized ALE/MsPacman-v5 environment.")
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        return

    state, info = env.reset()
    logging.info(f"State shape: {np.asarray(state).shape}")

    action_size = env.action_space.n
    agent = Agent(action_size=action_size)
    
    model_path = "dcql_model_episode_1000.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
    
    num_episodes = 100
    max_steps = 1000
    batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_target_frequency = 500
    save_frequency = 10
    step_count = 0
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        score = info.get('score', 0)
        lives = info.get('lives', 3)
        episode_steps = 0
        episode_reward = 0

        while episode_steps < max_steps:
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Phạt khi mất mạng
            if info.get('lives', 3) < lives:
                reward -= 50
            lives = info.get('lives', 3)

            episode_reward += reward
            agent.step(state, action, reward, next_state, done or truncated)
            
            state = next_state
            score = info.get('score', score)
            step_count += 1
            episode_steps += 1

            if step_count % update_target_frequency == 0:
                agent.update_target_network()

            if done or truncated or lives <= 0:
                break

        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % save_frequency == 0 and episode > 0:
            save_path = f"dcql_model_episode_{episode}.pth"
            agent.save_model(save_path)

        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        logging.info(f"Episode {episode + 1}/{num_episodes}, Score: {score}, Lives: {lives}, Epsilon: {epsilon:.3f}, Reward: {episode_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}")

    agent.save_model("dcql_model_episode_1000.pth")
    logging.info("Training completed.")
    env.close()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_dcqn()