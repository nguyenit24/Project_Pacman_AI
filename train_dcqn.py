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

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # For input 210x160x3
        self.fc2 = nn.Linear(512, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 255.0  # [B, H, W, C] to [B, C, H, W] and normalize
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = deque(maxlen=10000)
        self.minibatch_size = 64
        self.discount_factor = 0.99

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.minibatch_size:
            experiences = random.sample(self.memory, k=self.minibatch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1).to(self.device)

        Q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.discount_factor * Q_targets_next * (1 - dones))
        Q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())

    def save_model(self, model_path):
        try:
            torch.save(self.local_qnetwork.state_dict(), model_path)
            logging.info(f"Saved model to {model_path}")
        except Exception as e:
            logging.error(f"Failed to save model to {model_path}: {e}")

    def load_model(self, model_path):
        try:
            self.local_qnetwork.load_state_dict(torch.load(model_path))
            self.target_qnetwork.load_state_dict(torch.load(model_path))
            self.local_qnetwork.eval()
            self.target_qnetwork.eval()
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")

def train_dcqn():
    # Khởi tạo môi trường
    try:
        env = gym.make('ALE/MsPacman-v5', full_action_space=False, render_mode='rgb_array')
        logging.info("Successfully initialized ALE/MsPacman-v5 environment.")
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        return

    # Khởi tạo agent
    action_size = env.action_space.n
    agent = Agent(action_size=action_size)
    
    # Kiểm tra và tải mô hình nếu tồn tại
    model_path = "dcql_model_episode_1000.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
    
    # Tham số huấn luyện
    num_episodes = 1000
    max_steps = 1000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_target_frequency = 1000
    save_frequency = 100
    step_count = 0

    # Vòng lặp huấn luyện
    for episode in range(num_episodes):
        state, info = env.reset()
        score = info.get('score', 0)
        lives = info.get('lives', 3)
        episode_steps = 0
        episode_reward = 0

        while episode_steps < max_steps:
            # Chọn hành động
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Lưu kinh nghiệm
            agent.step(state, action, reward, next_state, done or truncated)
            
            # Cập nhật state và thông tin
            state = next_state
            score = info.get('score', score)
            lives = info.get('lives', lives)
            step_count += 1
            episode_steps += 1

            # Cập nhật target network
            if step_count % update_target_frequency == 0:
                agent.update_target_network()

            # Kiểm tra điều kiện dừng
            if done or truncated or lives <= 0:
                break

        # Giảm epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Lưu mô hình định kỳ
        if episode % save_frequency == 0 and episode > 0:
            save_path = f"dcql_model_episode_{episode}.pth"
            agent.save_model(save_path)

        # Ghi log tiến trình
        logging.info(f"Episode {episode + 1}/{num_episodes}, Score: {score}, Lives: {lives}, Epsilon: {epsilon:.3f}, Reward: {episode_reward:.2f}")

    # Lưu mô hình cuối cùng
    agent.save_model("dcql_model_episode_1000.pth")
    logging.info("Training completed.")
    env.close()

if __name__ == "__main__":
    # Đặt seed để tái lập kết quả
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_dcqn()