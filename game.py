import copy
import heapq
import math
import random
from collections import deque

import pygame

from board import boards
from ghost import Ghost
from logic import Pathfinder
from player import Player


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
        self.backtracking_times = []
        self.genetic_times = []
        self.level_menu = True
        self.menu = False
        self.paused = False  # Thêm trạng thái tạm dừng
        self.game_level = 1
        
        self.eaten_ghost = []
        self.ghost_speeds = []
        self.ghosts = []
        self.startup_counter_ghost = 0

        self.blinky_img = pygame.transform.scale(pygame.image.load("assets/ghost_images/red.png"), (45, 45))
        self.pinky_img = pygame.transform.scale(pygame.image.load("assets/ghost_images/pink.png"), (45, 45))

        try:
            self.menu_background = pygame.image.load("assets/bgmenu.jpg").convert()
            self.menu_background = pygame.transform.scale(self.menu_background, (self.WIDTH_SCREEN, self.HEIGHT))
        except pygame.error:
            self.menu_background = pygame.Surface((self.WIDTH_SCREEN, self.HEIGHT))
            self.menu_background.fill((0, 0, 0))

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

        pygame.draw.rect(self.screen, "gray", [self.WIDTH_SCREEN - 160, 10, 150, self.HEIGHT // 2 - 150], 0, 5)
        time_title = self.font.render("Time (s)", True, "white")
        self.screen.blit(time_title, (self.WIDTH_SCREEN - 150, 20))
        current_time_text = self.font.render(f"Current: {self.game_duration:.2f}", True, "white")
        self.screen.blit(current_time_text, (self.WIDTH_SCREEN - 150, 50))
        bfs_title = self.font.render("BFS:", True, "yellow")
        self.screen.blit(bfs_title, (self.WIDTH_SCREEN - 150, 80))
        for i, t in enumerate(self.bfs_times[:4]):
            bfs_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(bfs_time_text, (self.WIDTH_SCREEN - 150, 100 + i * 20))
        astar_title = self.font.render("A*:", True, "yellow")
        self.screen.blit(astar_title, (self.WIDTH_SCREEN - 150, 180))
        for i, t in enumerate(self.astar_times[:4]):
            astar_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(astar_time_text, (self.WIDTH_SCREEN - 150, 200 + i * 20))
        backtracking_title = self.font.render("Backtracking:", True, "yellow")
        self.screen.blit(backtracking_title, (self.WIDTH_SCREEN - 150, 280))
        for i, t in enumerate(self.backtracking_times[:4]):
            backtracking_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(backtracking_time_text, (self.WIDTH_SCREEN - 150, 300 + i * 20))
        genetic_title = self.font.render("Genetic:", True, "yellow")
        self.screen.blit(genetic_title, (self.WIDTH_SCREEN - 150, 380))
        for i, t in enumerate(self.genetic_times[:4]):
            genetic_time_text = self.font.render(f"{i+1}. {t:.2f}", True, "yellow")
            self.screen.blit(genetic_time_text, (self.WIDTH_SCREEN - 150, 400 + i * 20))

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
            if(ghost.in_box):
                target = (ghost.x_pos, ghost.y_pos - 100) 
            targets.append(target)
            
        return targets

    def reset(self):
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
        self.eaten_ghost = [False] * (self.game_level - 1)
        self.ghost_speeds = [1] * (self.game_level - 1)
        self.ghosts = []
        if self.game_level >= 2:
            self.ghosts.append(Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[0], self.blinky_img, 0, False, True, 0, self.level))
        if self.game_level >= 3:
            self.ghosts.append(Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[1], self.pinky_img, 1, False, True, 0, self.level))

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
                            selected_mode = self.run_backtracking
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()
                        elif event.key == pygame.K_5:
                            selected_mode = self.run_genetic
                            self.menu = False
                            self.paused = False
                            self.start_time = pygame.time.get_ticks()

            else:
                if selected_mode and not self.paused:
                    selected_mode()
                    if self.menu:
                        selected_mode = None
                elif self.paused:
                    self.draw_pause_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
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
        min_ghost_dist = float('inf') if not ghost_pos else min(self.logic.heuristic(current_pos, ghost) for ghost in ghost_pos)
        if min_ghost_dist <= 3:
            safe_dot = None
            for dot in dots:
                path = self.logic.rta_star_avoid_ghosts(current_pos, [dot], ghost_positions=ghost_pos)
                if path:
                    safe_dot = dot
                    break
            safe_point = None
            for i in range(len(self.level)):
                for j in range(len(self.level[0])):
                    if self.level[i][j] < 3:
                        pos = (i, j)
                        if all(self.logic.heuristic(pos, ghost) >= 5 for ghost in ghost_pos):
                            path = self.logic.rta_star_avoid_ghosts(current_pos, [pos], ghost_positions=ghost_pos)
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
                    if (0 <= new_pos[0] < len(self.level) and 
                        0 <= new_pos[1] < len(self.level[0]) and 
                        self.level[new_pos[0]][new_pos[1]] < 3):
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

            pygame.draw.circle(self.screen, "black", (center_x, center_y), 20, 2)
            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(center_x, center_y)

            if moving and not self.game_won:
                current_grid_pos = self.get_grid_pos(center_x, center_y)
                if self.path_to_target and not self.is_path_safe(self.path_to_target[self.current_target_index:], ghost_pos, danger_radius=3):
                    self.path_to_target = None
                    self.current_target_index = 0
                safe_dot, safe_point = self.find_dot_safe(current_grid_pos, ghost_pos)
                if self.path_to_target is None or self.current_target_index >= len(self.path_to_target) or (self.path_to_target and self.path_to_target[0] != current_grid_pos):
                    if safe_dot:
                        candidate = self.logic.rta_star_avoid_ghosts(current_grid_pos, [safe_dot], ghost_positions=ghost_pos)
                    elif safe_point:
                        candidate = self.logic.rta_star_avoid_ghosts(current_grid_pos, [safe_point], ghost_positions=ghost_pos)
                    else:
                        dots = self.find_dots()
                        candidate = self.logic.rta_star_avoid_ghosts(current_grid_pos, dots, ghost_positions=ghost_pos) if dots else None
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

    def run_BFS(self):
        run = True
        self.game_mode = 1
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

            # ghost_pos = [self.get_surrounding_positions(ghost.center_x, ghost.center_y) for ghost in self.ghosts]
            ghost_pos = []
            self.screen.fill("black")
            self.draw_board()
            self.draw_grid()
            center_x, center_y = self.player.get_center()

            self.game_won = all(1 not in row and 2 not in row for row in self.level)
            if self.game_won:
                if self.game_duration not in self.bfs_times:
                    self.bfs_times.insert(0, self.game_duration)

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
            self.flicker = (self.counter <= 3)
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
            if self.game_won and self.game_duration not in self.backtracking_times:
                self.backtracking_times.insert(0, self.game_duration)

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
                needs_new = (
                    not self.path_to_target or
                    self.current_target_index >= len(self.path_to_target) or
                    (self.path_to_target and self.path_to_target[0] != grid)
                )
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
            self.flicker = (self.counter <= 3)
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
                    move_fn = ghost.move_clyde if ghost.in_box or ghost.dead else (
                        ghost.move_blinky if i == 0 else ghost.move_pinky
                    )
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
            if self.game_won and self.game_duration not in self.genetic_times:
                self.genetic_times.insert(0, self.game_duration)

            self.player.draw(self.screen, self.counter)
            self.draw_misc()
            self.turns_allowed = self.check_position(cx, cy)

            if moving and not self.game_won and not self.game_over:
                grid = self.get_grid_pos(cx, cy)
                if self.path_to_target and not self.is_path_safe(self.path_to_target[self.current_target_index:], ghost_pos, danger_radius=4):
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
                        path = self.logic.genetic(
                            start=grid,
                            goals=[target],
                            ghost_positions=ghost_pos,
                            max_generations=150,
                            population_size=50
                        )
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
        Hiển thị menu chọn level riêng biệt.
        """
        self.screen.blit(self.menu_background, (0, 0))
        title_font = pygame.font.Font("freesansbold.ttf", 50)
        level_font = pygame.font.Font("freesansbold.ttf", 30)

        title_text = title_font.render("SELECT LEVEL", True, "yellow")
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.3))
        self.screen.blit(title_text, title_rect)

        levels = [
            {"text": "1. Level 1: No Ghosts", "key": pygame.K_1},
            {"text": "2. Level 2: 1 Ghost", "key": pygame.K_2},
            {"text": "3. Level 3: 2 Ghosts", "key": pygame.K_3}
        ]

        start_y = self.HEIGHT * 0.35
        spacing = self.HEIGHT * 0.1

        for i, level in enumerate(levels):
            level_text = level_font.render(level["text"], True, "white")
            level_rect = level_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            self.screen.blit(level_text, level_rect)

        pygame.display.flip()

    def draw_menu(self):
        """
        Hiển thị menu chính với các chế độ chơi.
        """
        self.screen.blit(self.menu_background, (0, 0))
        title_font = pygame.font.Font("freesansbold.ttf", 50)
        menu_font = pygame.font.Font("freesansbold.ttf", 30)

        title_text = title_font.render("PAC-MAN AI GAME", True, "yellow")
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.2))
        self.screen.blit(title_text, title_rect)

        modes = [
            {"text": "1. Manual Control", "key": pygame.K_1, "desc": "Use arrow keys to play"},
            {"text": "2. BFS Algorithm", "key": pygame.K_2, "desc": "AI with Breadth-First Search"},
            {"text": "3. A* Algorithm", "key": pygame.K_3, "desc": "AI with A* Search"},
            {"text": "4. Backtracking", "key": pygame.K_4, "desc": "AI with Backtracking"},
            {"text": "5. Genetic Algorithm", "key": pygame.K_5, "desc": "AI with Genetic Algorithm"}
        ]

        start_y = self.HEIGHT * 0.35
        spacing = self.HEIGHT * 0.1

        for i, mode in enumerate(modes):
            mode_text = menu_font.render(mode["text"], True, "white")
            mode_rect = mode_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            self.screen.blit(mode_text, mode_rect)

            desc_font = pygame.font.Font("freesansbold.ttf", 20)
            desc_text = desc_font.render(mode["desc"], True, "gray")
            desc_rect = desc_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing + 30))
            self.screen.blit(desc_text, desc_rect)

        pygame.display.flip()

    def draw_pause_menu(self):
        """
        Hiển thị menu tạm dừng với các tùy chọn: Tiếp tục, Quay về menu chế độ, Thoát game.
        """
        self.screen.blit(self.menu_background, (0, 0))
        title_font = pygame.font.Font("freesansbold.ttf", 50)
        option_font = pygame.font.Font("freesansbold.ttf", 30)

        title_text = title_font.render("PAUSED", True, "yellow")
        title_rect = title_text.get_rect(center=(self.WIDTH_SCREEN // 2, self.HEIGHT * 0.2))
        self.screen.blit(title_text, title_rect)

        options = [
            {"text": "1. Continue (P)", "key": pygame.K_1},
            {"text": "2. Back to Mode Menu", "key": pygame.K_2},
            {"text": "3. Exit Game", "key": pygame.K_3}
        ]

        start_y = self.HEIGHT * 0.35
        spacing = self.HEIGHT * 0.1

        for i, option in enumerate(options):
            option_text = option_font.render(option["text"], True, "white")
            option_rect = option_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            self.screen.blit(option_text, option_rect)

        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.run()