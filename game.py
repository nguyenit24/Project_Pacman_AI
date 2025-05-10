import os
import csv
import random
import heapq
import logging
import math
import copy
import pygame
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import ale_py

from board import boards
from ghost import Ghost
from logic import Pathfinder
from player import Player

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
        x = x.reshape(x.size(0), -1)  # Replaced view with reshape
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
class Game:
    def __init__(self):
        pygame.init()
        self.WIDTH = 900
        self.HEIGHT = 950
        self.WIDTH_SCREEN = 1100
        self.screen = pygame.display.set_mode([self.WIDTH_SCREEN, self.HEIGHT])
        self.timer = pygame.time.Clock()
        self.fps = 144
        self.font = pygame.font.Font("freesansbold.ttf", 20)
        self.level = copy.deepcopy(boards)
        self.logic = Pathfinder(self.level)
        self.color = "blue"
        self.PI = math.pi
        self.player = Player()
        self.counter = 0
        self.flicker = False
        self.turns_allowed = [False, False, False, False]
        self.direction_command = 0
        self.powerup = False
        self.power_counter = 0
        self.startup_counter = 0
        self.game_over = False
        self.game_won = False
        self.path_to_target = None
        self.current_target_index = 0
        self.game_mode = None
        self.game_duration = 0
        self.start_time = None
        self.bfs_times = []
        self.astar_times = []
        self.rtastar_times = []  # Added for RTA* times
        self.backtracking_times = []
        self.genetic_times = []
        self.level_menu = True
        self.menu = False
        self.paused = False
        self.game_level = 1
        self.eaten_ghost = []
        self.ghost_speeds = []
        self.ghosts = []
        self.startup_counter_ghost = 0
        self.data_saved = False
        self.blinky_img = pygame.transform.scale(pygame.image.load("assets/ghost_images/red.png"), (45, 45))
        self.pinky_img = pygame.transform.scale(pygame.image.load("assets/ghost_images/pink.png"), (45, 45))

        try:
            self.menu_background = pygame.image.load("assets/bgmenu.jpg").convert()
            self.menu_background = pygame.transform.scale(self.menu_background, (self.WIDTH_SCREEN, self.HEIGHT))
        except pygame.error:
            self.menu_background = pygame.Surface((self.WIDTH_SCREEN, self.HEIGHT))
            self.menu_background.fill((0, 0, 0))

        self.csv_file_path = os.path.join(os.path.dirname(__file__), "game_stats.csv")
        self.initialize_csv()
        self.dcql_times = []  # Thêm để lưu thời gian DCQL
        self.env = None
        self.dcql_agent = None
        if gym is not None and torch is not None:
            try:
                self.env = gym.make('ALE/MsPacman-v5', full_action_space=False, render_mode='rgb_array')
                self.dcql_agent = Agent(action_size=self.env.action_space.n)
                logging.info("Successfully initialized DCQL environment with ALE/MsPacman-v5.")
                model_path = "dcql_model_episode_1000.pth"
                if os.path.exists(model_path):
                    self.dcql_agent.load_model(model_path)
            except Exception as e:
                logging.error(f"Failed to initialize ALE/MsPacman-v5: {e}")
                self.env = None
                self.dcql_agent = None
                print("Warning: DCQL mode is unavailable due to environment setup issues. Other modes are still accessible.")
        else:
            logging.warning("Gymnasium or PyTorch not installed. DCQL mode will be unavailable.")
            print("Warning: DCQL mode is unavailable. Install gymnasium and torch to enable it.")
    def initialize_csv(self):
        """Initialize the CSV file with a header if it doesn't exist."""
        try:
            if not os.path.exists(self.csv_file_path):
                with open(self.csv_file_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Level", "Algorithm", "Duration", "Score", "Lives"])  # Added Algorithm column
                logging.info(f"Created CSV file at {self.csv_file_path} with header.")
            else:
                logging.info(f"CSV file already exists at {self.csv_file_path}.")
        except Exception as e:
            logging.error(f"Failed to create CSV file at {self.csv_file_path}: {e}")

    def save_game_data(self, level, algorithm, duration, score, lives):
        """Save game data to CSV file."""
        if self.data_saved:
            logging.info(f"Data already saved for this session: Level={level}, Algorithm={algorithm}")
            return
        try:
            if not isinstance(level, int) or level < 1:
                raise ValueError("Level must be a positive integer.")
            if not isinstance(algorithm, str):
                raise ValueError("Algorithm must be a string.")
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValueError("Duration must be a non-negative number.")
            if not isinstance(score, int) or score < 0:
                raise ValueError("Score must be a non-negative integer.")
            if not isinstance(lives, int) or lives < 0:
                raise ValueError("Lives must be a non-negative integer.")

            with open(self.csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([level, algorithm, f"{duration:.2f}", score, lives])
            logging.info(f"Data saved to {self.csv_file_path}: Level={level}, Algorithm={algorithm}, Duration={duration:.2f}, Score={score}, Lives={lives}")
            self.data_saved = True  # Mark data as saved
        except PermissionError:
            logging.error(f"Permission denied when writing to {self.csv_file_path}.")
        except ValueError as ve:
            logging.error(f"Invalid data provided: {ve}")
        except Exception as e:
            logging.error(f"Error saving data to {self.csv_file_path}: {e}")

    def draw_misc(self):
        score_text = self.font.render(f"Score: {self.player.score}", True, "white")
        self.screen.blit(score_text, (10, 920))
        level_text = self.font.render(f"Level: {self.game_level}", True, "white")
        self.screen.blit(level_text, (200, 920))
        if self.powerup:
            pygame.draw.circle(self.screen, "blue", (140, 930), 15)
        for i in range(self.player.lives):
            self.screen.blit(pygame.transform.scale(self.player.images[0], (30, 30)), (650 + i * 40, 915))
        if self.game_over:
            pygame.draw.rect(self.screen, "white", [50, 200, 800, 300], 0, 10)
            pygame.draw.rect(self.screen, "dark gray", [70, 220, 760, 260], 0, 10)
            gameover_text = self.font.render("Game over! Space bar to restart!", True, "red")
            self.screen.blit(gameover_text, (100, 300))
        if self.game_won:
            pygame.draw.rect(self.screen, "white", [50, 200, 800, 300], 0, 10)
            pygame.draw.rect(self.screen, "dark gray", [70, 220, 760, 260], 0, 10)
            gameover_text = self.font.render("Victory! Space bar to restart!", True, "green")
            self.screen.blit(gameover_text, (100, 300))

        pygame.draw.rect(self.screen, "gray", [self.WIDTH_SCREEN - 160, 10, 150, self.HEIGHT - 50], 0, 5)
        time_title = self.font.render("Time (s)", True, "white")
        self.screen.blit(time_title, (self.WIDTH_SCREEN - 150, 20))
        current_time_text = self.font.render(f"Current: {self.game_duration:.2f}", True, "white")
        self.screen.blit(current_time_text, (self.WIDTH_SCREEN - 150, 50))

        # BFS Times
        bfs_title = self.font.render("BFS:", True, "yellow")
        self.screen.blit(bfs_title, (self.WIDTH_SCREEN - 150, 80))
        for i, t in enumerate(self.bfs_times[:4]):
            bfs_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(bfs_time_text, (self.WIDTH_SCREEN - 150, 100 + i * 20))

        # A* Times
        astar_title = self.font.render("A*:", True, "yellow")
        self.screen.blit(astar_title, (self.WIDTH_SCREEN - 150, 180))
        for i, t in enumerate(self.astar_times[:4]):
            astar_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(astar_time_text, (self.WIDTH_SCREEN - 150, 200 + i * 20))

        # Real-Time A* Times
        rtastar_title = self.font.render("RT A*:", True, "yellow")
        self.screen.blit(rtastar_title, (self.WIDTH_SCREEN - 150, 280))
        for i, t in enumerate(self.rtastar_times[:4]):
            rtastar_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(rtastar_time_text, (self.WIDTH_SCREEN - 150, 300 + i * 20))

        # Backtracking Times
        backtracking_title = self.font.render("Backtracking:", True, "yellow")
        self.screen.blit(backtracking_title, (self.WIDTH_SCREEN - 150, 380))
        for i, t in enumerate(self.backtracking_times[:4]):
            backtracking_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(backtracking_time_text, (self.WIDTH_SCREEN - 150, 400 + i * 20))

        # Genetic Times
        genetic_title = self.font.render("Genetic:", True, "yellow")
        self.screen.blit(genetic_title, (self.WIDTH_SCREEN - 150, 480))
        for i, t in enumerate(self.genetic_times[:4]):
            genetic_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(genetic_time_text, (self.WIDTH_SCREEN - 150, 500 + i * 20))

        # DCQL Times
        dcql_title = self.font.render("DCQL:", True, "yellow")
        self.screen.blit(dcql_title, (self.WIDTH_SCREEN - 150, 580))
        for i, t in enumerate(self.dcql_times[:4]):
            dcql_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(dcql_time_text, (self.WIDTH_SCREEN - 150, 600 + i * 20))

    def draw_board(self):
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] == 1:
                    pygame.draw.circle(self.screen, "white", (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 4)
                if self.level[i][j] == 2 and not self.flicker:
                    pygame.draw.circle(self.screen, "white", (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 10)
                if self.level[i][j] == 3:
                    pygame.draw.line(self.screen, self.color, (j * num2 + (0.5 * num2), i * num1), (j * num2 + (0.5 * num2), i * num1 + num1), 3)
                if self.level[i][j] == 4:
                    pygame.draw.line(self.screen, self.color, (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)
                if self.level[i][j] == 5:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 + (0.5 * num1)), num2, num1], 0, self.PI / 2, 3)
                if self.level[i][j] == 6:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 + (num2 * 0.5)), (i * num1 + (0.5 * num1)), num2, num1], self.PI / 2, self.PI, 3)
                if self.level[i][j] == 7:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 + (num2 * 0.5)), (i * num1 - (0.4 * num1)), num2, num1], self.PI, 3 * self.PI / 2, 3)
                if self.level[i][j] == 8:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 - (0.4 * num1)), num2, num1], 3 * self.PI / 2, 2 * self.PI, 3)
                if self.level[i][j] == 9:
                    pygame.draw.line(self.screen, "white", (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)

    def check_collisions(self):
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        center_x, center_y = self.player.get_center()
        if 0 < self.player.x < 870:
            if self.level[center_y // num1][center_x // num2] == 1:
                self.level[center_y // num1][center_x // num2] = 0
                self.player.score += 10
            if self.level[center_y // num1][center_x // num2] == 2:
                self.level[center_y // num1][center_x // num2] = 0
                self.player.score += 50
                self.powerup = True
                self.power_counter = 0
        return self.powerup, self.power_counter

    def check_position(self, centerx, centery):
        turns = [False, False, False, False]
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        num3 = 15
        if centerx // 30 < 29:
            if self.player.direction == 0:
                if self.level[centery // num1][(centerx - num3) // num2] < 3:
                    turns[1] = True
            if self.player.direction == 1:
                if self.level[centery // num1][(centerx + num3) // num2] < 3:
                    turns[0] = True
            if self.player.direction == 2:
                if self.level[(centery + num3) // num1][centerx // num2] < 3:
                    turns[3] = True
            if self.player.direction == 3:
                if self.level[(centery - num3) // num1][centerx // num2] < 3:
                    turns[2] = True

            if self.player.direction in (2, 3):
                if 12 <= centerx % num2 <= 18:
                    if self.level[(centery + num3) // num1][centerx // num2] < 3:
                        turns[3] = True
                    if self.level[(centery - num3) // num1][centerx // num2] < 3:
                        turns[2] = True
                if 12 <= centery % num1 <= 18:
                    if self.level[centery // num1][(centerx - num2) // num2] < 3:
                        turns[1] = True
                    if self.level[centery // num1][(centerx + num2) // num2] < 3:
                        turns[0] = True
            if self.player.direction in (0, 1):
                if 12 <= centerx % num2 <= 18:
                    if self.level[(centery + num1) // num1][centerx // num2] < 3:
                        turns[3] = True
                    if self.level[(centery - num1) // num1][centerx // num2] < 3:
                        turns[2] = True
                if 12 <= centery % num1 <= 18:
                    if self.level[centery // num1][(centerx - num3) // num2] < 3:
                        turns[1] = True
                    if self.level[centery // num1][(centerx + num3) // num2] < 3:
                        turns[0] = True
        else:
            turns[0] = True
            turns[1] = True
        return turns

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_grid_pos(self, x, y):
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        return (y // num1, x // num2)

    def find_dots(self):
        dots = []
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] in (1, 2):
                    dots.append((i, j))
        return dots

    def get_direction_from_path(self, current_pos, next_pos):
        if next_pos[1] > current_pos[1]:
            return 0
        elif next_pos[1] < current_pos[1]:
            return 1
        elif next_pos[0] < current_pos[0]:
            return 2
        elif next_pos[0] > current_pos[0]:
            return 3
        return self.player.direction

    def is_at_center(self, x, y, grid_pos):
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        target_x = grid_pos[1] * num2 + num2 // 2
        target_y = grid_pos[0] * num1 + num1 // 2
        return abs(x + 23 - target_x) < self.player.speed and abs(y + 24 - target_y) < self.player.speed

    def check_ghost_collisions(self):
        if self.game_level == 1:
            return False
        player_rect = pygame.Rect(self.player.x, self.player.y, 45, 45)
        if not self.powerup:
            for i, ghost in enumerate(self.ghosts):
                ghost_rect = ghost.draw()
                if player_rect.colliderect(ghost_rect) and not ghost.dead:
                    if self.player.lives > 0:
                        self.player.lives -= 1
                        self.startup_counter = 0
                        self.powerup = False
                        self.power_counter = 0
                        self.player.x = 450
                        self.player.y = 663
                        self.player.direction = 0
                        self.direction_command = 0
                        self.path_to_target = None
                        self.current_target_index = 0
                        for g in self.ghosts:
                            g.dead = False
                        return True
                    else:
                        self.game_over = True
                        return True
        if self.powerup:
            for i, ghost in enumerate(self.ghosts):
                ghost_rect = ghost.draw()
                if player_rect.colliderect(ghost_rect) and self.eaten_ghost[i] and not ghost.dead:
                    if self.player.lives > 0:
                        self.powerup = False
                        self.power_counter = 0
                        self.player.lives -= 1
                        self.startup_counter = 0
                        self.player.x = 450
                        self.player.y = 663
                        self.player.direction = 0
                        self.direction_command = 0
                        for g in self.ghosts:
                            g.dead = False
                        return True
                    else:
                        self.game_over = True
                        return True
                if player_rect.colliderect(ghost_rect) and not ghost.dead and not self.eaten_ghost[i]:
                    ghost.dead = True
                    self.eaten_ghost[i] = True
                    self.player.score += (2 ** self.eaten_ghost.count(True)) * 100
        return False

    def get_surrounding_positions(self, center_x, center_y, distance=3):
        grid_x, grid_y = self.get_grid_pos(center_x, center_y)
        surrounding_positions = []
        height = len(self.level)
        width = len(self.level[0])
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                if (dx**2 + dy**2) ** 0.5 <= distance:
                    new_y = grid_y + dy
                    new_x = grid_x + dx
                    if 0 <= new_y < height and 0 <= new_x < width and self.level[new_y][new_x] < 3:
                        surrounding_positions.append((new_y, new_x))
        return surrounding_positions

    def get_targets(self):
        if self.game_level == 1:
            return []
        player_x, player_y = self.player.x, self.player.y
        ghost_positions = [(ghost.x_pos, ghost.y_pos) for ghost in self.ghosts]
        if player_x < 450:
            runaway_x = 900
        else:
            runaway_x = 0
        if player_y < 450:
            runaway_y = 900
        else:
            runaway_y = 0
        return_target = (380, 400)
        targets = []
        for i, ghost in enumerate(self.ghosts):
            if self.powerup:
                if not ghost.dead:
                    target = (runaway_x, runaway_y)
                else:
                    target = return_target
            else:
                if not ghost.dead:
                    target = (player_x, player_y)
                else:
                    target = return_target
            if ghost.in_box:
                target = (ghost.x_pos, ghost.y_pos - 100)
            targets.append(target)

        return targets

    def reset(self):
        """Reset game state without saving data."""
        self.powerup = False
        self.power_counter = 0
        self.startup_counter = 0
        self.player = Player()
        self.direction_command = 0
        self.player.score = 0
        self.player.lives = 3
        self.level = copy.deepcopy(boards)
        self.logic = Pathfinder(self.level)
        self.game_over = False
        self.game_won = False
        self.path_to_target = None
        self.current_target_index = 0
        self.game_duration = 0
        self.start_time = None
        self.startup_counter_ghost = 0
        self.data_saved = False  # Reset data saved flag
        self.eaten_ghost = [False] * (self.game_level - 1)
        self.ghost_speeds = [1] * (self.game_level - 1)
        self.ghosts = []
        if self.game_level >= 2:
            self.ghosts.append(Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[0], self.blinky_img, 0, False, True, 0, self.level))
        if self.game_level >= 3:
            self.ghosts.append(Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[1], self.pinky_img, 1, False, True, 0, self.level))
    def run_deep_convolutional_q_learning(self):
        if self.env is None or self.dcql_agent is None:
            logging.error("Gymnasium or PyTorch not available, cannot run DCQL.")
            self.menu = True
            self.level_menu = True
            return

        run = True
        self.game_mode = 6
        moving = False
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        episodes = 1000
        max_steps = 1000
        update_target_frequency = 1000
        save_frequency = 100
        step_count = 0

        for episode in range(episodes):
            state, info = self.env.reset()
            self.player.x = 450
            self.player.y = 663
            self.player.direction = 0
            self.player.score = info.get('score', 0)
            self.player.lives = info.get('lives', 3)
            self.level = copy.deepcopy(boards)
            self.powerup = False
            self.power_counter = 0
            self.startup_counter = 0
            self.game_over = False
            self.game_won = False
            self.start_time = pygame.time.get_ticks()
            self.data_saved = False
            episode_steps = 0

            while not self.game_over and episode_steps < max_steps:
                self.timer.tick(self.fps)
                if self.paused:
                    self.draw_pause_menu()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            run = False
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_p:
                                self.paused = False
                            elif event.key == pygame.K_1:
                                self.paused = False
                            elif event.key == pygame.K_2:
                                self.reset()
                                self.menu = True
                                self.level_menu = True
                                self.paused = False
                                run = False
                                return
                            elif event.key == pygame.K_3:
                                pygame.quit()
                                return
                    pygame.display.flip()
                    continue

                if not self.game_won:
                    current_time = pygame.time.get_ticks()
                    self.game_duration = (current_time - self.start_time) / 1000.0

                if self.counter < 19:
                    self.counter += 1
                    if self.counter > 3:
                        self.flicker = False
                else:
                    self.counter = 0
                    self.flicker = True

                if self.powerup and self.power_counter < 600:
                    self.power_counter += 1
                elif self.powerup and self.power_counter >= 600:
                    self.power_counter = 0
                    self.powerup = False

                if self.startup_counter < 30 and not self.game_over and not self.game_won:
                    moving = False
                    self.startup_counter += 1
                else:
                    moving = True

                if self.startup_counter_ghost < 60:
                    self.startup_counter_ghost += 1
                else:
                    targets = self.get_targets()
                    for i, ghost in enumerate(self.ghosts):
                        ghost.target = targets[i] if i < len(targets) else ""
                        ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                        if ghost.in_box:
                            ghost.dead = False
                            ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                        else:
                            ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                        ghost.center_x = ghost.x_pos + 22
                        ghost.center_y = ghost.y_pos + 22
                        ghost.turns, ghost.in_box = ghost.check_collisions()

                self.screen.fill("black")
                self.draw_board()
                center_x, center_y = self.player.get_center()
                self.turns_allowed = self.check_position(center_x, center_y)

                action = self.dcql_agent.act(state, epsilon)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.dcql_agent.step(state, action, reward, next_state, done or truncated)
                state = next_state
                step_count += 1
                episode_steps += 1

                new_score = info.get('score', self.player.score)
                new_lives = info.get('lives', self.player.lives)

                if new_score > self.player.score:
                    self.player.score = new_score
                    num1 = (self.HEIGHT - 50) // 32
                    num2 = self.WIDTH // 30
                    grid_x = center_x // num2
                    grid_y = center_y // num1
                    if 0 <= grid_y < len(self.level) and 0 <= grid_x < len(self.level[0]):
                        if self.level[grid_y][grid_x] == 1:
                            self.level[grid_y][grid_x] = 0
                        elif self.level[grid_y][grid_x] == 2:
                            self.level[grid_y][grid_x] = 0
                            self.powerup = True
                            self.power_counter = 0

                if new_lives < self.player.lives:
                    self.player.lives = new_lives
                    self.startup_counter = 0
                    self.player.x = 450
                    self.player.y = 663
                    self.player.direction = 0
                    self.powerup = False
                    self.power_counter = 0

                action_to_direction = {0: None, 1: 2, 2: 0, 3: 1, 4: 3}
                if moving and not self.game_won:
                    direction = action_to_direction.get(action)
                    if direction is not None and self.turns_allowed[direction]:
                        self.player.direction = direction
                        self.player.move(self.turns_allowed)

                self.game_won = all(1 not in row and 2 not in row for row in self.level)
                if self.game_won:
                    if self.game_duration not in self.dcql_times:
                        self.dcql_times.insert(0, self.game_duration)
                        logging.info(f"DCQL completed Level {self.game_level} in {self.game_duration:.2f}s")
                        self.save_game_data(self.game_level, "DCQL", self.game_duration, self.player.score, self.player.lives)

                if self.player.lives <= 0 or done or truncated:
                    self.game_over = True
                    moving = False
                    self.startup_counter = 0
                    if not self.data_saved:
                        logging.info(f"DCQL game over at Level {self.game_level}")
                        self.save_game_data(self.game_level, "DCQL", self.game_duration, self.player.score, self.player.lives)

                self.player.draw(self.screen, self.counter)
                self.draw_misc()

                for ghost in self.ghosts:
                    ghost.draw()
                for ghost in self.ghosts:
                    ghost.target = ()

                self.powerup, self.power_counter = self.check_collisions()
                if self.check_ghost_collisions():
                    moving = False
                    self.startup_counter = 0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        self.menu = True
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = True
                        if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            run = False
                            return

                self.draw_ghost_radii(ghost_radius=3)
                pygame.display.flip()

                if step_count % update_target_frequency == 0:
                    self.dcql_agent.update_target_network()

                if episode % save_frequency == 0 and episode > 0:
                    self.dcql_agent.save_model(f"dcql_model_episode_{episode}.pth")

                if done or truncated or self.game_over or self.game_won:
                    break

            epsilon = max(epsilon_min, epsilon_decay * epsilon)
            logging.info(f"Episode {episode + 1}/{episodes} completed. Epsilon: {epsilon:.3f}, Score: {self.player.score}, Lives: {self.player.lives}")

        self.dcql_agent.save_model("dcql_model_episode_1000.pth")
        self.menu = True
        self.level_menu = True

# Cập nhật run để thêm tùy chọn DCQL
    def run(self):
        run = True
        selected_mode = None
        while run:
            self.timer.tick(self.fps)
            self.screen.fill("black")

            if self.level_menu:
                self.draw_level_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.start_time is not None:
                            self.game_duration = (pygame.time.get_ticks() - self.start_time) / 1000.0
                            algorithm = {0: "Manual", 1: "BFS", 2: "A*", 3: "Backtracking", 4: "Genetic", 5: "RTA*", 6: "DCQL"}.get(self.game_mode, "Unknown")
                            logging.info(f"Saving data on quit: Level={self.game_level}, Algorithm={algorithm}, Duration={self.game_duration:.2f}, Score={self.player.score}, Lives={self.player.lives}")
                            self.save_game_data(self.game_level, algorithm, self.game_duration, self.player.score, self.player.lives)
                        run = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            self.game_level = 1
                            self.reset()
                            self.level_menu = False
                            self.menu = True
                        elif event.key == pygame.K_2:
                            self.game_level = 2
                            self.reset()
                            self.level_menu = False
                            self.menu = True
                        elif event.key == pygame.K_3:
                            self.game_level = 3
                            self.reset()
                            self.level_menu = False
                            self.menu = True

            elif self.menu:
                self.draw_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.start_time is not None:
                            self.game_duration = (pygame.time.get_ticks() - self.start_time) / 1000.0
                            algorithm = {0: "Manual", 1: "BFS", 2: "A*", 3: "Backtracking", 4: "Genetic", 5: "RTA*", 6: "DCQL"}.get(self.game_mode, "Unknown")
                            logging.info(f"Saving data on quit: Level={self.game_level}, Algorithm={algorithm}, Duration={self.game_duration:.2f}, Score={self.player.score}, Lives={self.player.lives}")
                            self.save_game_data(self.game_level, algorithm, self.game_duration, self.player.score, self.player.lives)
                        run = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            selected_mode = self.run_manual
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_2:
                            selected_mode = self.run_BFS
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_3:
                            selected_mode = self.run_A_star
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_4:
                            selected_mode = self.run_RTA_star
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_5:
                            selected_mode = self.run_backtracking
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_6:
                            selected_mode = self.run_genetic
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_7:
                            selected_mode = self.run_deep_convolutional_q_learning
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_8:
                            self.show_statistics()

            else:
                if selected_mode and not self.paused:
                    selected_mode()
                    if self.menu:
                        selected_mode = None
                elif self.paused:
                    self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.start_time is not None:
                            self.game_duration = (pygame.time.get_ticks() - self.start_time) / 1000.0
                            algorithm = {0: "Manual", 1: "BFS", 2: "A*", 3: "Backtracking", 4: "Genetic", 5: "RTA*", 6: "DCQL"}.get(self.game_mode, "Unknown")
                            logging.info(f"Saving data on quit: Level={self.game_level}, Algorithm={algorithm}, Duration={self.game_duration:.2f}, Score={self.player.score}, Lives={self.player.lives}")
                            self.save_game_data(self.game_level, algorithm, self.game_duration, self.player.score, self.player.lives)
                        run = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False

            pygame.display.flip()
        pygame.quit()
    

    def run_manual(self):
        run = True
        self.game_mode = 0
        moving = False

        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0

            if self.counter < 19:
                self.counter += 1
                if self.counter > 3:
                    self.flicker = False
            else:
                self.counter = 0
                self.flicker = True

            if self.powerup and self.power_counter < 600:
                self.power_counter += 1
            elif self.powerup and self.power_counter >= 600:
                self.power_counter = 0
                self.powerup = False

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    if ghost.in_box:
                        ghost.dead = False
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                    else:
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()

            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            center_x, center_y = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.bfs_times:  # Manual mode uses bfs_times as placeholder
                    self.bfs_times.insert(0, self.game_duration)
                    logging.info(f"Manual completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "Manual", self.game_duration, self.player.score, self.player.lives)
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0
                if not self.data_saved:
                    logging.info(f"Manual game over at Level {self.game_level}")
                    self.save_game_data(self.game_level, "Manual", self.game_duration, self.player.score, self.player.lives)
            pygame.draw.circle(self.screen, "black", (center_x, center_y), 20, 2)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(center_x, center_y)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_RIGHT:
                        self.direction_command = 0
                    if event.key == pygame.K_LEFT:
                        self.direction_command = 1
                    if event.key == pygame.K_UP:
                        self.direction_command = 2
                    if event.key == pygame.K_DOWN:
                        self.direction_command = 3
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            if moving and not self.game_won:
                if self.direction_command == 0 and self.turns_allowed[0]:
                    self.player.direction = 0
                if self.direction_command == 1 and self.turns_allowed[1]:
                    self.player.direction = 1
                if self.direction_command == 2 and self.turns_allowed[2]:
                    self.player.direction = 2
                if self.direction_command == 3 and self.turns_allowed[3]:
                    self.player.direction = 3
                self.player.move(self.turns_allowed)

            for ghost in self.ghosts:
                ghost.draw()
            for ghost in self.ghosts:
                ghost.target = ()

            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def draw_path(self):
        if self.path_to_target:
            num1 = (self.HEIGHT - 50) // 32
            num2 = self.WIDTH // 30
            points = [(pos[1] * num2 + num2 // 2, pos[0] * num1 + num1 // 2) for pos in self.path_to_target]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (255, 182, 193), False, points, 4)

    def print_movement_debug(self, current_grid_pos, next_grid_pos=None):
        at_center = self.is_at_center(self.player.x, self.player.y, current_grid_pos)
        print("====== MOVEMENT DEBUG ======")
        print(f"Player pixel pos: ({self.player.x}, {self.player.y})")
        print(f"Player grid pos: {current_grid_pos}")
        if next_grid_pos:
            print(f"Next grid pos: {next_grid_pos}")
        print(f"is_at_center: {at_center}")
        print(f"turns_allowed: {self.turns_allowed}")
        print(f"Current target index: {self.current_target_index}")
        print(f"path_to_target: {self.path_to_target}")
        print("============================")

    def draw_grid(self):
        num1 = (self.HEIGHT - 50) // 32
        num2 = self.WIDTH // 30
        rows = len(self.level)
        cols = len(self.level[0])
        for col in range(cols + 1):
            x = col * num2
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.HEIGHT), 1)
            if col < cols:
                text = self.font.render(str(col), True, (200, 200, 200))
                self.screen.blit(text, (x + num2 // 2 - text.get_width() // 2, 0))
        for row in range(rows + 1):
            y = row * num1
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.WIDTH, y), 1)
            if row < rows:
                text = self.font.render(str(row), True, (200, 200, 200))
                self.screen.blit(text, (0, y + num1 // 2 - text.get_height() // 2))

    def draw_ghost_radii(self, ghost_radius=3):
        num2 = self.WIDTH // 30
        pixel_radius = ghost_radius * num2
        for ghost in self.ghosts:
            pygame.draw.circle(self.screen, (255, 0, 0), (ghost.center_x, ghost.center_y), pixel_radius, 2)

    def is_ghost_near(self, current_grid_pos, threshold=2):
        if self.game_level == 1:
            return False
        for ghost in self.ghosts:
            ghost_grid = self.get_grid_pos(ghost.center_x, ghost.center_y)
            if self.logic.heuristic(current_grid_pos, ghost_grid) < threshold:
                return True
        return False

    def find_dot_safe(self, current_pos, ghost_pos):
        dots = self.find_dots()
        if not dots:
            return None, None
        min_ghost_dist = float("inf") if not ghost_pos else min(self.logic.heuristic(current_pos, ghost) for ghost in ghost_pos)
        if min_ghost_dist <= 3:
            safe_dot = None
            for dot in dots:
                path = self.logic.A_star(current_pos, [dot], ghost_positions=ghost_pos)
                if path:
                    safe_dot = dot
                    break
            safe_point = None
            for i in range(len(self.level)):
                for j in range(len(self.level[0])):
                    if self.level[i][j] < 3:
                        pos = (i, j)
                        if all(self.logic.heuristic(pos, ghost) >= 5 for ghost in ghost_pos):
                            path = self.logic.A_star(current_pos, [pos], ghost_positions=ghost_pos)
                            if path:
                                safe_point = pos
                                break
                if safe_point:
                    break
            return safe_dot, safe_point
        closest_dot = min(dots, key=lambda dot: self.logic.heuristic(current_pos, dot))
        return closest_dot, None

    def predict_ghost_positions(self, ghosts, steps=4):
        if self.game_level == 1:
            return []
        predicted_positions = []
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        for ghost in ghosts:
            current_pos = self.get_grid_pos(ghost.center_x, ghost.center_y)
            predicted_positions.append(current_pos)
            current_direction = ghost.direction
            for _ in range(steps):
                possible_moves = []
                for dx, dy in directions:
                    new_pos = (current_pos[0] + dy, current_pos[1] + dx)
                    if 0 <= new_pos[0] < len(self.level) and 0 <= new_pos[1] < len(self.level[0]) and self.level[new_pos[0]][new_pos[1]] < 3:
                        possible_moves.append(new_pos)
                if possible_moves:
                    if current_direction == 0:
                        next_pos = min(possible_moves, key=lambda pos: pos[1])
                    elif current_direction == 1:
                        next_pos = max(possible_moves, key=lambda pos: pos[1])
                    elif current_direction == 2:
                        next_pos = max(possible_moves, key=lambda pos: pos[0])
                    else:
                        next_pos = min(possible_moves, key=lambda pos: pos[0])
                    predicted_positions.append(next_pos)
                    current_pos = next_pos
        return list(set(predicted_positions))

    def is_path_safe(self, path, ghost_positions, danger_radius=4):
        for pos in path:
            for ghost_pos in ghost_positions:
                if self.heuristic(pos, ghost_pos) <= danger_radius:
                    return False
        return True

    def run_A_star(self):
        run = True
        self.game_mode = 2
        moving = False

        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0

            if self.counter < 19:
                self.counter += 1
                if self.counter > 3:
                    self.flicker = False
            else:
                self.counter = 0
                self.flicker = True
            if self.powerup and self.power_counter < 600:
                self.power_counter += 1
            elif self.powerup and self.power_counter >= 600:
                self.power_counter = 0
                self.powerup = False

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    if ghost.in_box or ghost.dead:
                        ghost.dead = False
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                    else:
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()

            ghost_pos = self.predict_ghost_positions(self.ghosts, steps=4)

            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            center_x, center_y = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.astar_times:
                    self.astar_times.insert(0, self.game_duration)
                    logging.info(f"A* completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "A*", self.game_duration, self.player.score, self.player.lives)
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0
                if not self.data_saved:
                    logging.info(f"A* game over at Level {self.game_level}")
                    self.save_game_data(self.game_level, "A*", self.game_duration, self.player.score, self.player.lives)

            pygame.draw.circle(self.screen, "black", (center_x, center_y), 20, 2)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(center_x, center_y)

            if moving and not self.game_won:
                current_grid_pos = self.get_grid_pos(center_x, center_y)
                if self.path_to_target and not self.is_path_safe(self.path_to_target[self.current_target_index :], ghost_pos, danger_radius=3):
                    self.path_to_target = None
                    self.current_target_index = 0
                safe_dot, safe_point = self.find_dot_safe(current_grid_pos, ghost_pos)
                if self.path_to_target is None or self.current_target_index >= len(self.path_to_target) or (self.path_to_target and self.path_to_target[0] != current_grid_pos):
                    if safe_dot:
                        candidate = self.logic.A_star(current_grid_pos, [safe_dot], ghost_positions=ghost_pos)
                    elif safe_point:
                        candidate = self.logic.A_star(current_grid_pos, [safe_point], ghost_positions=ghost_pos)
                    else:
                        dots = self.find_dots()
                        candidate = self.logic.A_star(current_grid_pos, dots, ghost_positions=ghost_pos) if dots else None
                    if candidate:
                        self.path_to_target = candidate
                        self.current_target_index = 1
                    else:
                        self.path_to_target = None
                if self.path_to_target and self.current_target_index < len(self.path_to_target):
                    next_grid_pos = self.path_to_target[self.current_target_index]
                    if self.is_at_center(self.player.x, self.player.y, current_grid_pos):
                        if current_grid_pos == next_grid_pos:
                            self.current_target_index += 1
                        else:
                            self.player.direction = self.get_direction_from_path(current_grid_pos, next_grid_pos)
                            if self.turns_allowed[self.player.direction]:
                                self.player.move(self.turns_allowed)
                    else:
                        if self.turns_allowed[self.player.direction]:
                            self.player.move(self.turns_allowed)

            for ghost in self.ghosts:
                ghost.draw()
            for ghost in self.ghosts:
                ghost.target = ()

            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def run_RTA_star(self):
        run = True
        self.game_mode = 5
        moving = False

        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0

            if self.counter < 19:
                self.counter += 1
                if self.counter > 3:
                    self.flicker = False
            else:
                self.counter = 0
                self.flicker = True

            if self.powerup and self.power_counter < 600:
                self.power_counter += 1
            elif self.powerup and self.power_counter >= 600:
                self.power_counter = 0
                self.powerup = False

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    if ghost.in_box:
                        ghost.dead = False
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                    else:
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()
            ghost_pos = self.predict_ghost_positions(self.ghosts, steps=4)
            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            center_x, center_y = self.player.get_center()
            self.turns_allowed = self.check_position(center_x, center_y)
            current_grid_pos = self.get_grid_pos(center_x, center_y)

            # Kiểm tra điều kiện thắng
            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.rtastar_times:  # Sửa từ astar_times thành rtastar_times
                    self.rtastar_times.insert(0, self.game_duration)
                    logging.info(f"RTA* completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "RTA*", self.game_duration, self.player.score, self.player.lives)
            # Vẽ marker (tùy chọn)
            pygame.draw.circle(self.screen, "black", (center_x, center_y), 20, 2)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()

            # Xác định mục tiêu dựa trên vị trí các dot an toàn (nếu có)
            safe_dot, safe_point = self.find_dot_safe(current_grid_pos, ghost_pos)
            if safe_dot:
                goal = safe_dot
            elif safe_point:
                goal = safe_point
            else:
                dots = self.find_dots()
                goal = dots[0] if dots else current_grid_pos

            # Tính bước đi tiếp theo sử dụng RTA*: chỉ tính toán các trạng thái lân cận
            next_grid_pos = self.logic.rta_star_realtime(current_grid_pos, goal, ghost_positions=ghost_pos)

            # Vì RTA* tính 1 bước tại thời điểm hiện tại, chúng ta tạo "path" với current và next
            if not self.path_to_target or self.path_to_target[0] != current_grid_pos:
                self.path_to_target = [current_grid_pos, next_grid_pos]
                self.current_target_index = 1

            # Nếu ở giữa ô, cập nhật hướng dựa trên next_grid_pos
            if moving and not self.game_won:
                if self.is_at_center(self.player.x, self.player.y, current_grid_pos):
                    if current_grid_pos == next_grid_pos:
                        self.current_target_index += 1
                    else:
                        self.player.direction = self.get_direction_from_path(current_grid_pos, next_grid_pos)
                        if self.turns_allowed[self.player.direction]:
                            self.player.move(self.turns_allowed)
                else:
                    if self.turns_allowed[self.player.direction]:
                        self.player.move(self.turns_allowed)

            for ghost in self.ghosts:
                ghost.draw()
            for ghost in self.ghosts:
                ghost.target = ()

            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def run_BFS(self):
        run = True
        self.game_mode = 1
        moving = False
        self.start_time = pygame.time.get_ticks()  # Khởi tạo start_time
        self.data_saved = False  # Reset data_saved

        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0  # Tính thời gian

            if self.counter < 19:
                self.counter += 1
                if self.counter > 3:
                    self.flicker = False
            else:
                self.counter = 0
                self.flicker = True

            if self.powerup and self.power_counter < 600:
                self.power_counter += 1
            elif self.powerup and self.power_counter >= 600:
                self.power_counter = 0
                self.powerup = False

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    if ghost.in_box or ghost.dead:
                        ghost.dead = False
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                    else:
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()

            ghost_pos = []
            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            center_x, center_y = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won and not self.data_saved:
                if not any(abs(self.game_duration - t) < 0.01 for t in self.bfs_times): 
                    self.bfs_times.insert(0, self.game_duration)
                    logging.info(f"BFS completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "BFS", self.game_duration, self.player.score, self.player.lives)
                    

            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0
                if not self.data_saved:
                    logging.info(f"BFS game over at Level {self.game_level}")
                    self.save_game_data(self.game_level, "BFS", self.game_duration, self.player.score, self.player.lives)
                   
            pygame.draw.circle(self.screen, "black", (center_x, center_y), 20, 2)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(center_x, center_y)

            if moving and not self.game_won:
                current_grid_pos = self.get_grid_pos(center_x, center_y)
                if self.path_to_target is None or self.current_target_index >= len(self.path_to_target) or (self.path_to_target and self.path_to_target[0] != current_grid_pos):
                    dots = self.find_dots()
                    if dots:
                        self.path_to_target = self.logic.bfs(current_grid_pos, dots, ghost_pos)
                        self.current_target_index = 0
                    else:
                        self.path_to_target = None
                if self.path_to_target and self.current_target_index < len(self.path_to_target):
                    next_grid_pos = self.path_to_target[self.current_target_index]
                    if self.is_at_center(self.player.x, self.player.y, current_grid_pos):
                        if current_grid_pos == next_grid_pos:
                            self.current_target_index += 1
                        else:
                            self.player.direction = self.get_direction_from_path(current_grid_pos, next_grid_pos)
                            if self.turns_allowed[self.player.direction]:
                                self.player.move(self.turns_allowed)
                    else:
                        if self.turns_allowed[self.player.direction]:
                            self.player.move(self.turns_allowed)

            for ghost in self.ghosts:
                ghost.draw()
            for ghost in self.ghosts:
                ghost.target = ()

            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def run_backtracking(self):
        run = True
        self.game_mode = 3
        moving = False
        depth_limit = 50
        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0

            self.counter = (self.counter + 1) % 20
            self.flicker = self.counter <= 3
            if self.powerup:
                self.power_counter += 1
                if self.power_counter >= 600:
                    self.powerup, self.power_counter = False, 0

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    if ghost.in_box or ghost.dead:
                        ghost.dead = False
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                    else:
                        ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky() if i == 0 else ghost.move_pinky()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()

            ghost_pos = [self.get_surrounding_positions(ghost.center_x, ghost.center_y) for ghost in self.ghosts]

            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            cx, cy = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.backtracking_times:
                    self.backtracking_times.insert(0, self.game_duration)
                    logging.info(f"Backtracking completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "Backtracking", self.game_duration, self.player.score, self.player.lives)
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0
                if not self.data_saved:
                    logging.info(f"Backtracking game over at Level {self.game_level}")
                    self.save_game_data(self.game_level, "Backtracking", self.game_duration, self.player.score, self.player.lives)

            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(cx, cy)

            if moving and not self.game_won and not self.game_over:
                grid = self.get_grid_pos(cx, cy)
                if self.path_to_target and self.current_target_index < len(self.path_to_target):
                    nxt = self.path_to_target[self.current_target_index]
                    if self.is_at_center(self.player.x, self.player.y, grid):
                        if grid == nxt:
                            self.current_target_index += 1
                        else:
                            d = self.get_direction_from_path(grid, nxt)
                            if self.turns_allowed[d]:
                                self.player.direction = d
                                self.player.move(self.turns_allowed)
                            else:
                                self.path_to_target = None
                    else:
                        if self.turns_allowed[self.player.direction]:
                            self.player.move(self.turns_allowed)
                needs_new = not self.path_to_target or self.current_target_index >= len(self.path_to_target) or (self.path_to_target and self.path_to_target[0] != grid)
                if needs_new:
                    dots = self.find_dots()
                    if dots:
                        try:
                            path = self.logic.backtracking(grid, dots, ghost_positions=ghost_pos, depth_limit=depth_limit)
                        except RecursionError:
                            path = None
                        if not path:
                            try:
                                path = self.logic.backtracking(grid, dots, ghost_positions=None, depth_limit=depth_limit)
                            except RecursionError:
                                path = None
                        if path and len(path) > 1:
                            self.path_to_target = path
                            self.current_target_index = 1
                        else:
                            self.path_to_target = None
                            self.current_target_index = 0
                    else:
                        self.path_to_target = None
                        self.current_target_index = 0
                if not self.path_to_target:
                    dirs = [i for i, ok in enumerate(self.turns_allowed) if ok]
                    if dirs:
                        d = random.choice(dirs)
                        self.player.direction = d
                        self.player.move(self.turns_allowed)

            for ghost in self.ghosts:
                ghost.draw()
            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def run_genetic(self):
        run = True
        self.game_mode = 4
        moving = False

        while run:
            self.timer.tick(self.fps)
            if self.paused:
                self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                        elif event.key == pygame.K_1:
                            self.paused = False
                        elif event.key == pygame.K_2:
                            self.reset()
                            self.menu = True
                            self.level_menu = True
                            self.paused = False
                            run = False
                            return
                        elif event.key == pygame.K_3:
                            pygame.quit()
                            return
                pygame.display.flip()
                continue

            if not self.game_won and not self.game_over:
                current_time = pygame.time.get_ticks()
                self.game_duration = (current_time - self.start_time) / 1000.0

            self.counter = (self.counter + 1) % 20
            self.flicker = self.counter <= 3
            if self.powerup:
                self.power_counter += 1
                if self.power_counter >= 600:
                    self.powerup, self.power_counter = False, 0

            if self.startup_counter < 30 and not self.game_over and not self.game_won:
                moving = False
                self.startup_counter += 1
            else:
                moving = True

            if self.startup_counter_ghost < 60:
                self.startup_counter_ghost += 1
            else:
                targets = self.get_targets()
                for i, ghost in enumerate(self.ghosts):
                    ghost.target = targets[i] if i < len(targets) else ""
                    ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                    move_fn = ghost.move_clyde if ghost.in_box or ghost.dead else (ghost.move_blinky if i == 0 else ghost.move_pinky)
                    if ghost.in_box:
                        ghost.dead = False
                    ghost.x_pos, ghost.y_pos, ghost.direction = move_fn()
                    ghost.center_x = ghost.x_pos + 22
                    ghost.center_y = ghost.y_pos + 22
                    ghost.turns, ghost.in_box = ghost.check_collisions()

            ghost_pos = self.predict_ghost_positions(self.ghosts, steps=4)

            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            cx, cy = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.genetic_times:
                    self.genetic_times.insert(0, self.game_duration)
                    logging.info(f"Genetic completed Level {self.game_level} in {self.game_duration:.2f}s")
                    self.save_game_data(self.game_level, "Genetic", self.game_duration, self.player.score, self.player.lives)
            if self.player.lives <= 0:
                self.game_over = True
                moving = False
                self.startup_counter = 0
                if not self.data_saved:
                    logging.info(f"Genetic game over at Level {self.game_level}")
                    self.save_game_data(self.game_level, "Genetic", self.game_duration, self.player.score, self.player.lives)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(cx, cy)

            if moving and not self.game_won and not self.game_over:
                grid = self.get_grid_pos(cx, cy)
                if self.path_to_target and not self.is_path_safe(self.path_to_target[self.current_target_index :], ghost_pos, danger_radius=3):
                    self.path_to_target = None
                    self.current_target_index = 0
                if self.path_to_target and self.current_target_index < len(self.path_to_target):
                    nxt = self.path_to_target[self.current_target_index]
                    if self.is_at_center(self.player.x, self.player.y, grid):
                        if grid == nxt:
                            self.current_target_index += 1
                        else:
                            d = self.get_direction_from_path(grid, nxt)
                            if self.turns_allowed[d]:
                                self.player.direction = d
                                self.player.move(self.turns_allowed)
                            else:
                                self.path_to_target = None
                                self.current_target_index = 0
                    else:
                        if self.turns_allowed[self.player.direction]:
                            self.player.move(self.turns_allowed)
                if not self.path_to_target or self.current_target_index >= len(self.path_to_target) or (self.path_to_target and self.path_to_target[0] != grid):
                    safe_dot, safe_point = self.find_dot_safe(grid, ghost_pos)
                    target = safe_dot if safe_dot else safe_point
                    if target:
                        path = self.logic.genetic(start=grid, goals=[target], ghost_positions=ghost_pos, max_generations=150, population_size=50)
                        if path and len(path) > 1 and self.is_path_safe(path, ghost_pos, danger_radius=3):
                            self.path_to_target = path
                            self.current_target_index = 1
                        else:
                            self.path_to_target = None
                            self.current_target_index = 0
                    else:
                        self.path_to_target = None
                        self.current_target_index = 0

            for ghost in self.ghosts:
                ghost.draw()
            self.powerup, self.power_counter = self.check_collisions()
            if self.check_ghost_collisions():
                moving = False
                self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    self.menu = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = True
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        self.menu = True
                        self.level_menu = True
                        run = False
                        return

            self.draw_path()
            self.draw_ghost_radii(ghost_radius=3)
            pygame.display.flip()

    def draw_level_menu(self):
        """
        Hiển thị menu chọn level với giao diện đẹp hơn.
        """
        # Tạo nền với hiệu ứng mờ
        overlay = pygame.Surface((self.WIDTH_SCREEN, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(self.menu_background, (0, 0))
        self.screen.blit(overlay, (0, 0))

        # Font và màu sắc
        title_font = pygame.font.Font("freesansbold.ttf", 60)
        level_font = pygame.font.Font("freesansbold.ttf", 36)
        shadow_offset = 3

        # Tiêu đề
        title_text = title_font.render("SELECT LEVEL", True, (255, 255, 0))
        title_shadow = title_font.render("SELECT LEVEL", True, (100, 100, 0))
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.25))
        self.screen.blit(title_shadow, (title_rect.x + shadow_offset, title_rect.y + shadow_offset))
        self.screen.blit(title_text, title_rect)

        # Các level
        levels = [{"text": "1. Level 1: No Ghosts", "key": pygame.K_1, "color": (255, 255, 255)}, {"text": "2. Level 2: 1 Ghost", "key": pygame.K_2, "color": (255, 200, 200)}, {"text": "3. Level 3: 2 Ghosts", "key": pygame.K_3, "color": (200, 200, 255)}]

        start_y = self.HEIGHT * 0.4
        spacing = self.HEIGHT * 0.12
        button_width = 400
        button_height = 60

        for i, level in enumerate(levels):
            # Vẽ nút nền
            button_rect = pygame.Rect(self.WIDTH_SCREEN // 2 - button_width // 2, start_y + i * spacing - button_height // 2, button_width, button_height)
            pygame.draw.rect(self.screen, (50, 50, 50, 200), button_rect, 0, 10)
            pygame.draw.rect(self.screen, level["color"], button_rect, 2, 10)

            # Vẽ văn bản
            level_text = level_font.render(level["text"], True, level["color"])
            level_shadow = level_font.render(level["text"], True, (50, 50, 50))
            text_rect = level_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            self.screen.blit(level_shadow, (text_rect.x + shadow_offset, text_rect.y + shadow_offset))
            self.screen.blit(level_text, text_rect)

        pygame.display.flip()

    def show_statistics(self):
        """Hiển thị thống kê từ file CSV trên màn hình, so sánh các thuật toán theo level với thời gian ở millisecond."""
        try:
            with open(self.csv_file_path, mode="r") as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Bỏ qua dòng tiêu đề
                if header is None:
                    print("CSV file is empty.")
                    logging.warning(f"CSV file {self.csv_file_path} is empty.")
                    return
                data = []
                for i, row in enumerate(reader):
                    # Bỏ qua dòng trống hoặc không đủ cột
                    if len(row) < 5:
                        logging.warning(f"Skipping invalid row {i+2} in CSV: {row}")
                        continue
                    # Kiểm tra row[0] có phải số nguyên hợp lệ
                    try:
                        int(row[0])
                    except ValueError:
                        logging.warning(f"Skipping row {i+2} with invalid level: {row}")
                        continue
                    # Kiểm tra Duration, Score, Lives có hợp lệ
                    try:
                        float(row[2])  # Duration
                        int(row[3])    # Score
                        int(row[4])    # Lives
                    except ValueError:
                        logging.warning(f"Skipping row {i+2} with invalid data: {row}")
                        continue
                    data.append(row)
        except FileNotFoundError:
            print("No statistics available.")
            logging.warning(f"CSV file {self.csv_file_path} not found.")
            return

        # Lọc dữ liệu cho level hiện tại
        level_data = [row for row in data if int(row[0]) == self.game_level]
        if not level_data:
            print(f"No statistics available for Level {self.game_level}.")
            logging.warning(f"No data found for Level {self.game_level}.")
            return

        # Nhóm dữ liệu theo thuật toán
        algorithms = sorted(list(set(row[1] for row in level_data)))  # Sắp xếp để nhất quán
        durations = {algo: [] for algo in algorithms}
        scores = {algo: [] for algo in algorithms}
        lives = {algo: [] for algo in algorithms}

        for row in level_data:
            algo = row[1]
            durations[algo].append(float(row[2]) * 1000)  # Chuyển sang millisecond
            scores[algo].append(int(row[3]))
            lives[algo].append(int(row[4]))

        # Tính giá trị trung bình
        avg_durations = [np.mean(durations[algo]) if durations[algo] else 0 for algo in algorithms]
        avg_scores = [np.mean(scores[algo]) if scores[algo] else 0 for algo in algorithms]
        avg_lives = [np.mean(lives[algo]) if lives[algo] else 0 for algo in algorithms]

        # Tạo biểu đồ so sánh
        plt.figure(figsize=(10, 8))  # Kích thước gọn gàng

        # Biểu đồ Duration (ms)
        plt.subplot(3, 1, 1)
        bars = plt.bar(algorithms, avg_durations, color="blue", alpha=0.7, width=0.3)
        plt.title(f"Average Duration (ms) - Level {self.game_level}", fontsize=10)
        plt.xlabel("Algorithm", fontsize=8)
        plt.ylabel("Duration (ms)", fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", ha="center", va="bottom", fontsize=7)

        # Biểu đồ Score
        plt.subplot(3, 1, 2)
        bars = plt.bar(algorithms, avg_scores, color="green", alpha=0.7, width=0.3)
        plt.title(f"Average Score - Level {self.game_level}", fontsize=10)
        plt.xlabel("Algorithm", fontsize=8)
        plt.ylabel("Score", fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.0f}", ha="center", va="bottom", fontsize=7)

        # Biểu đồ Lives
        plt.subplot(3, 1, 3)
        bars = plt.bar(algorithms, avg_lives, color="red", alpha=0.7, width=0.3)
        plt.title(f"Average Remaining Lives - Level {self.game_level}", fontsize=10)
        plt.xlabel("Algorithm", fontsize=8)
        plt.ylabel("Lives", fontsize=8)
        plt.xticks(rotation=45, fontsize=8)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout(pad=2.0)  # Khoảng cách giữa các biểu đồ
        plt.show()  # Hiển thị biểu đồ trên màn hình
        logging.info(f"Displayed statistics for Level {self.game_level}: {algorithms}")

    def draw_menu(self):
        """
        Hiển thị menu chính với giao diện tối ưu cho 6 thuật toán và thống kê.
        """
        # Tạo nền với hiệu ứng gradient
        for y in range(self.HEIGHT):
            color = (0, 0, y // 4)
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH_SCREEN, y))
        self.screen.blit(self.menu_background, (0, 0), special_flags=pygame.BLEND_MULT)

        # Font và màu sắc
        title_font = pygame.font.Font("freesansbold.ttf", 60)
        menu_font = pygame.font.Font("freesansbold.ttf", 32)
        desc_font = pygame.font.Font("freesansbold.ttf", 20)
        shadow_offset = 2

        # Tiêu đề
        title_text = title_font.render("PAC-MAN AI", True, (255, 255, 0))
        title_shadow = title_font.render("PAC-MAN AI", True, (100, 100, 0))
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.1))
        self.screen.blit(title_shadow, (title_rect.x + shadow_offset, title_rect.y + shadow_offset))
        self.screen.blit(title_text, title_rect)

        # Các chế độ chơi
        modes = [
            {"text": "1. Manual", "key": pygame.K_1, "desc": "Control with arrow keys", "color": (255, 255, 255)},
            {"text": "2. BFS", "key": pygame.K_2, "desc": "Breadth-First Search AI", "color": (200, 255, 200)},
            {"text": "3. A*", "key": pygame.K_3, "desc": "A* Search AI", "color": (200, 200, 255)},
            {"text": "4. Real Time A*", "key": pygame.K_4, "desc": "Real-Time A* AI", "color": (255, 200, 255)},
            {"text": "5. Backtracking", "key": pygame.K_5, "desc": "Backtracking AI", "color": (255, 200, 200)},
            {"text": "6. Genetic", "key": pygame.K_6, "desc": "Genetic Algorithm AI", "color": (200, 255, 255)},
            {"text": "7. DCQL", "key": pygame.K_7, "desc": "Deep Convolutional Q-Learning", "color": (255, 255, 200)},
            {"text": "8. Statistics", "key": pygame.K_8, "desc": "View performance stats", "color": (255, 255, 0)},
        ]

        start_y = self.HEIGHT * 0.2
        spacing = self.HEIGHT * 0.1
        button_width = 400
        button_height = 60

        for i, mode in enumerate(modes):
            # Vẽ nút nền
            button_rect = pygame.Rect(self.WIDTH_SCREEN // 2 - button_width // 2, start_y + i * spacing - button_height // 2, button_width, button_height)
            pygame.draw.rect(self.screen, (50, 50, 50, 200), button_rect, 0, 10)
            pygame.draw.rect(self.screen, mode["color"], button_rect, 2, 10)

            # Vẽ văn bản chính
            mode_text = menu_font.render(mode["text"], True, mode["color"])
            mode_shadow = menu_font.render(mode["text"], True, (50, 50, 50))
            text_rect = mode_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing - 10))
            self.screen.blit(mode_shadow, (text_rect.x + shadow_offset, text_rect.y + shadow_offset))
            self.screen.blit(mode_text, text_rect)

            # Vẽ mô tả
            desc_text = desc_font.render(mode["desc"], True, (200, 200, 200))
            desc_rect = desc_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing + 20))
            self.screen.blit(desc_text, desc_rect)

        pygame.display.flip()

    def draw_pause_menu(self):
        """
        Hiển thị menu tạm dừng với giao diện đẹp hơn.
        """
        # Tạo hiệu ứng mờ cho nền
        overlay = pygame.Surface((self.WIDTH_SCREEN, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        # Font và màu sắc
        title_font = pygame.font.Font("freesansbold.ttf", 60)
        option_font = pygame.font.Font("freesansbold.ttf", 36)
        shadow_offset = 3

        # Tiêu đề
        title_text = title_font.render("PAUSED", True, (255, 255, 0))
        title_shadow = title_font.render("PAUSED", True, (100, 100, 0))
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.25))
        self.screen.blit(title_shadow, (title_rect.x + shadow_offset, title_rect.y + shadow_offset))
        self.screen.blit(title_text, title_rect)

        # Các tùy chọn
        options = [{"text": "1. Continue (P)", "key": pygame.K_1, "color": (255, 255, 255)}, {"text": "2. Back to Mode Menu", "key": pygame.K_2, "color": (255, 200, 200)}, {"text": "3. Exit Game", "key": pygame.K_3, "color": (200, 200, 255)}]

        start_y = self.HEIGHT * 0.4
        spacing = self.HEIGHT * 0.12
        button_width = 400
        button_height = 60

        for i, option in enumerate(options):
            # Vẽ nút nền
            button_rect = pygame.Rect(self.WIDTH_SCREEN // 2 - button_width // 2, start_y + i * spacing - button_height // 2, button_width, button_height)
            pygame.draw.rect(self.screen, (50, 50, 50, 200), button_rect, 0, 10)
            pygame.draw.rect(self.screen, option["color"], button_rect, 2, 10)

            # Vẽ văn bản
            option_text = option_font.render(option["text"], True, option["color"])
            option_shadow = option_font.render(option["text"], True, (50, 50, 50))
            text_rect = option_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            self.screen.blit(option_shadow, (text_rect.x + shadow_offset, text_rect.y + shadow_offset))
            self.screen.blit(option_text, text_rect)

        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    # game.game_level = 3
    # game.reset()
    game.run()
