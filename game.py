import copy
import pygame
import math
from collections import deque
import heapq
from board import boards
from player import Player
from ghost import Ghost
class Game:
    def __init__(self):
        pygame.init()
        self.WIDTH = 900
        self.HEIGHT = 950
        self.WIDTH_SCREEN = 1100
        self.screen = pygame.display.set_mode([self.WIDTH_SCREEN, self.HEIGHT])
        self.timer = pygame.time.Clock()
        self.fps = 144
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.level = copy.deepcopy(boards)
        self.color = 'blue'
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
        
        self.eaten_ghost = [False, False, False, False]  
        self.ghost_speeds = [1, 1, 1, 1]  
        self.startup_counter_ghost = 0  
        
        # Tải hình ảnh cho ghosts
        self.blinky_img = pygame.transform.scale(pygame.image.load('assets/ghost_images/red.png'), (45, 45))
        self.pinky_img = pygame.transform.scale(pygame.image.load('assets/ghost_images/pink.png'), (45, 45))
        self.inky_img = pygame.transform.scale(pygame.image.load('assets/ghost_images/blue.png'), (45, 45))
        self.clyde_img = pygame.transform.scale(pygame.image.load('assets/ghost_images/orange.png'), (45, 45))

        # Khởi tạo 4 ghosts
        self.ghosts = [
            Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[0], self.blinky_img, 0, False, True, 0, self.level),  
            Ghost(400, 400, (self.player.x, self.player.y), self.ghost_speeds[1], self.pinky_img, 0, False, True, 1, self.level),   
            Ghost(500, 400, (self.player.x, self.player.y), self.ghost_speeds[2], self.inky_img, 0, False, True, 2, self.level),    
            Ghost(450, 450, (self.player.x, self.player.y), self.ghost_speeds[3], self.clyde_img, 0, False, True, 3, self.level)    
        ]

        try:
            self.menu_background = pygame.image.load('assets/bgmenu.jpg').convert()  
            self.menu_background = pygame.transform.scale(self.menu_background, (self.WIDTH_SCREEN, self.HEIGHT)) 
        except pygame.error:
            self.menu_background = pygame.Surface((self.WIDTH_SCREEN, self.HEIGHT)) 
            self.menu_background.fill((0, 0, 0))
        
        
    def draw_menu(self):
        self.screen.blit(self.menu_background, (0, 0))
        margin = self.WIDTH_SCREEN * 0.1 
        maze_top = self.HEIGHT * 0.05
        maze_bottom = self.HEIGHT * 0.95 
        
        pygame.draw.line(self.screen, (0, 0, 255), (margin, maze_top), (self.WIDTH_SCREEN- margin, maze_top), 5)  
        pygame.draw.line(self.screen, (0, 0, 255), (margin, maze_bottom), (self.WIDTH_SCREEN- margin, maze_bottom), 5)  #
        pygame.draw.line(self.screen, (0, 0, 255), (margin * 2, maze_top), (margin * 2, maze_bottom), 5)
        pygame.draw.line(self.screen, (0, 0, 255), (self.WIDTH_SCREEN - margin * 2, maze_top), (self.WIDTH_SCREEN - margin * 2, maze_bottom), 5) 
        
        
        dot_spacing = self.WIDTH_SCREEN // 12  
        for x in range(int(margin + dot_spacing), int(self.WIDTH_SCREEN - margin), dot_spacing):
            pygame.draw.circle(self.screen, (255, 255, 255), (x, int(maze_top + 50)), 5)  
            pygame.draw.circle(self.screen, (255, 255, 255), (x, int(maze_bottom - 50)), 5)  
        
        
        pacman_size = self.WIDTH_SCREEN * 0.05  
        pacman_x = margin
        pacman_y = self.HEIGHT * 0.1 
        pygame.draw.circle(self.screen, (255, 255, 0), (int(pacman_x), int(pacman_y)), int(pacman_size))  
        pygame.draw.polygon(self.screen, (0, 0, 20), [
            (int(pacman_x), int(pacman_y)),
            (int(pacman_x + pacman_size * 1.5), int(pacman_y - pacman_size * 0.5)),
            (int(pacman_x + pacman_size * 1.5), int(pacman_y + pacman_size * 0.5))
        ])  
        

        modes = [
            {'text': 'Manual Control', 'key': pygame.K_1, 'desc': 'Use arrow keys to play'},
            {'text': 'BFS Algorithm', 'key': pygame.K_2, 'desc': 'AI with Breadth-First Search'},
            {'text': 'A* Algorithm', 'key': pygame.K_3, 'desc': 'AI with A* Search'}
        ]
        
        menu_font = pygame.font.Font('freesansbold.ttf', 40)
        desc_font = pygame.font.Font('freesansbold.ttf', 20)
        start_y = self.HEIGHT * 0.35  # Bắt đầu từ 35% chiều cao
        spacing = self.HEIGHT * 0.15  # Khoảng cách là 15% chiều cao
        
        for i, mode in enumerate(modes):
            mode_text = menu_font.render(f"{i+1}. {mode['text']}", True, 'blue')
            mode_rect = mode_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing))
            
            desc_text = desc_font.render(mode['desc'], True, 'gray')
            desc_rect = desc_text.get_rect(center=(self.WIDTH_SCREEN // 2, start_y + i * spacing + 40))
            
            keys = pygame.key.get_pressed()
            if keys[mode['key']]:
                mode_text = menu_font.render(f"{i+1}. {mode['text']}", True, 'yellow')
            
            self.screen.blit(mode_text, mode_rect)
            self.screen.blit(desc_text, desc_rect)
        
        pygame.display.flip()
    def draw_misc(self):
        score_text = self.font.render(f'Score: {self.player.score}', True, 'white')
        self.screen.blit(score_text, (10, 920))
        if self.powerup:
            pygame.draw.circle(self.screen, 'blue', (140, 930), 15)
        for i in range(self.player.lives):
            self.screen.blit(pygame.transform.scale(self.player.images[0], (30, 30)), (650 + i * 40, 915))
        if self.game_over:
            pygame.draw.rect(self.screen, 'white', [50, 200, 800, 300], 0, 10)
            pygame.draw.rect(self.screen, 'dark gray', [70, 220, 760, 260], 0, 10)
            gameover_text = self.font.render('Game over! Space bar to restart!', True, 'red')
            self.screen.blit(gameover_text, (100, 300))
        if self.game_won:
         
            pygame.draw.rect(self.screen, 'white', [50, 200, 800, 300], 0, 10)
            pygame.draw.rect(self.screen, 'dark gray', [70, 220, 760, 260], 0, 10)
            gameover_text = self.font.render('Victory! Space bar to restart!', True, 'green')
            self.screen.blit(gameover_text, (100, 300))

        # Cập nhật bảng thời gian
        pygame.draw.rect(self.screen, 'gray', [self.WIDTH_SCREEN - 160, 10, 150, self.HEIGHT//2 -150], 0, 5)
        time_title = self.font.render('Time (s)', True, 'white')
        self.screen.blit(time_title, (self.WIDTH_SCREEN - 150, 20))
        
        current_time_text = self.font.render(f'Current: {self.game_duration:.2f}', True, 'white')
        self.screen.blit(current_time_text, (self.WIDTH_SCREEN - 150, 50))
        
        bfs_title = self.font.render('BFS:', True, 'yellow')
        self.screen.blit(bfs_title, (self.WIDTH_SCREEN - 150, 80))
        for i, t in enumerate(self.bfs_times[:4]):  # Giới hạn 4 dòng
            bfs_time_text = self.font.render(f'{i+1}. {t:.2f}', True, 'yellow')
            self.screen.blit(bfs_time_text, (self.WIDTH_SCREEN - 150, 100 + i * 20))
        
        astar_title = self.font.render('A*:', True, 'yellow')
        self.screen.blit(astar_title, (self.WIDTH_SCREEN - 150, 180))
        for i, t in enumerate(self.astar_times[:4]):  # Giới hạn 4 dòng
            astar_time_text = self.font.render(f'{i+1}. {t:.2f}', True, 'yellow')
            self.screen.blit(astar_time_text, (self.WIDTH_SCREEN - 150, 200 + i * 20))

    def draw_board(self):
        num1 = ((self.HEIGHT - 50) // 32)
        num2 = (self.WIDTH // 30)
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] == 1:
                    pygame.draw.circle(self.screen, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 4)
                if self.level[i][j] == 2 and not self.flicker:
                    pygame.draw.circle(self.screen, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 10)
                if self.level[i][j] == 3:
                    pygame.draw.line(self.screen, self.color, (j * num2 + (0.5 * num2), i * num1),
                                    (j * num2 + (0.5 * num2), i * num1 + num1), 3)
                if self.level[i][j] == 4:
                    pygame.draw.line(self.screen, self.color, (j * num2, i * num1 + (0.5 * num1)),
                                    (j * num2 + num2, i * num1 + (0.5 * num1)), 3)
                if self.level[i][j] == 5:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 + (0.5 * num1)), num2, num1],
                                    0, self.PI / 2, 3)
                if self.level[i][j] == 6:
                    pygame.draw.arc(self.screen, self.color,
                                    [(j * num2 + (num2 * 0.5)), (i * num1 + (0.5 * num1)), num2, num1], self.PI / 2, self.PI, 3)
                if self.level[i][j] == 7:
                    pygame.draw.arc(self.screen, self.color, [(j * num2 + (num2 * 0.5)), (i * num1 - (0.4 * num1)), num2, num1], self.PI,
                                    3 * self.PI / 2, 3)
                if self.level[i][j] == 8:
                    pygame.draw.arc(self.screen, self.color,
                                    [(j * num2 - (num2 * 0.4)) - 2, (i * num1 - (0.4 * num1)), num2, num1], 3 * self.PI / 2,
                                    2 * self.PI, 3)
                if self.level[i][j] == 9:
                    pygame.draw.line(self.screen, 'white', (j * num2, i * num1 + (0.5 * num1)),
                                    (j * num2 + num2, i * num1 + (0.5 * num1)), 3)

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
        num2 = (self.WIDTH // 30)
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

    def bfs(self, start, target_list, ghost_pos):
        if not target_list:
            return None
        queue = deque([(start, [])])
        visited = set([start])
        
        # Làm phẳng ghost_pos thành một tập hợp các vị trí duy nhất
        ghost_positions = set()
        for ghost_area in ghost_pos:  
            ghost_positions.update(ghost_area)  
        
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        while queue:
            (row, col), path = queue.popleft()
            
            if (row, col) in target_list:
                return path + [(row, col)]
            
            for dx, dy in directions:
                new_row, new_col = row + dx, col + dy
                # Kiểm tra điều kiện: trong lưới, không phải tường, chưa thăm, và không nằm trong ghost_positions
                if (0 <= new_row < len(self.level) and 
                    0 <= new_col < len(self.level[0]) and
                    self.level[new_row][new_col] < 3 and 
                    (new_row, new_col) not in visited and 
                    (new_row, new_col) not in ghost_positions):
                    queue.append(((new_row, new_col), path + [(row, col)]))
                    visited.add((new_row, new_col))
        
        return None

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star(self, start, target_list):
        if not target_list:
            return None
        open_set = [(0, start, [])]
        heapq.heapify(open_set)
        visited = set()
        g_scores = {start: 0}
        
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        while open_set:
            f_score, current, path = heapq.heappop(open_set)
            
            if current in target_list:
                return path + [current]
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for dx, dy in directions:
                new_row, new_col = current[0] + dx, current[1] + dy
                new_pos = (new_row, new_col)
                
                if (0 <= new_row < len(self.level) and 0 <= new_col < len(self.level[0]) and
                    self.level[new_row][new_col] < 3 and new_pos not in visited):
                    
                    tentative_g_score = g_scores[current] + 1
                    if new_pos not in g_scores or tentative_g_score < g_scores[new_pos]:
                        g_scores[new_pos] = tentative_g_score
                        h_score = min(self.heuristic(new_pos, target) for target in target_list)
                        f_score = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score, new_pos, path + [current]))
        
        return None

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
        
        player_rect = pygame.Rect(self.player.x, self.player.y, 45, 45)
        if not self.powerup:
            for i, ghost in enumerate(self.ghosts):
                ghost_rect = ghost.draw()  # 
                if player_rect.colliderect(ghost_rect) and not ghost.dead:
                    if self.player.lives > 0:
                        self.player.lives -= 1
                        self.startup_counter = 0
                        self.powerup = False
                        self.power_counter = 0
                        # Đặt lại vị trí Pac-Man
                        self.player.x = 450
                        self.player.y = 663
                        self.player.direction = 0
                        self.direction_command = 0
                        # Đặt lại vị trí và trạng thái các ghost
                        self.ghosts[0].x_pos, self.ghosts[0].y_pos = 56, 58
                        self.ghosts[0].direction = 0
                        self.ghosts[1].x_pos, self.ghosts[1].y_pos = 440, 388
                        self.ghosts[1].direction = 2
                        self.ghosts[2].x_pos, self.ghosts[2].y_pos = 440, 438
                        self.ghosts[2].direction = 2
                        self.ghosts[3].x_pos, self.ghosts[3].y_pos = 440, 438
                        self.ghosts[3].direction = 2
                        self.eaten_ghost = [False, False, False, False]
                        for ghost in self.ghosts:
                            ghost.dead = False
                        return True  # Trả về True để báo hiệu cần reset
                    else:
                        self.game_over = True
                        return True  # Trả về True để báo hiệu game over
        
        # Kiểm tra va chạm khi có powerup
        if self.powerup:
            for i, ghost in enumerate(self.ghosts):
                ghost_rect = ghost.draw()
                # Trường hợp ghost đã bị ăn nhưng chưa chết (không xảy ra trong logic hiện tại, bỏ qua)
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
                        self.ghosts[0].x_pos, self.ghosts[0].y_pos = 56, 58
                        self.ghosts[0].direction = 0
                        self.ghosts[1].x_pos, self.ghosts[1].y_pos = 440, 388
                        self.ghosts[1].direction = 2
                        self.ghosts[2].x_pos, self.ghosts[2].y_pos = 440, 438
                        self.ghosts[2].direction = 2
                        self.ghosts[3].x_pos, self.ghosts[3].y_pos = 440, 438
                        self.ghosts[3].direction = 2
                        self.eaten_ghost = [False, False, False, False]
                        for ghost in self.ghosts:
                            ghost.dead = False
                        return True
                    else:
                        self.game_over = True
                        return True
                
                # Trường hợp ăn ghost khi có powerup
                if player_rect.colliderect(ghost_rect) and not ghost.dead and not self.eaten_ghost[i]:
                    ghost.dead = True
                    self.eaten_ghost[i] = True
                    # Cộng điểm theo số ghost đã ăn (100, 200, 400, 800)
                    self.player.score += (2 ** self.eaten_ghost.count(True)) * 100
        
        return False  # Không cần reset nếu không có va chạm nghiêm trọng
    def get_surrounding_positions(self, center_x, center_y, distance=2):
        # Chuyển đổi tọa độ pixel thành tọa độ lưới
        grid_x, grid_y = self.get_grid_pos(center_x, center_y)
        
        # Tạo danh sách các vị trí xung quanh
        surrounding_positions = []
        
        # Kích thước lưới
        height = len(self.level)
        width = len(self.level[0])
        
        # Duyệt qua các ô trong phạm vi ±distance (2 ô)
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                new_y = grid_y + dy
                new_x = grid_x + dx
                # Kiểm tra xem vị trí có nằm trong lưới và không phải tường không
                if 0 <= new_y < height and 0 <= new_x < width and self.level[new_y][new_x] < 3:
                    surrounding_positions.append((new_y, new_x))
        
        return surrounding_positions
    def get_targets(self):
        # Lấy vị trí của Pac-Man
        player_x, player_y = self.player.x, self.player.y
        
        # Lấy vị trí của các ghost từ danh sách self.ghosts
        blink_x, blink_y = self.ghosts[0].x_pos, self.ghosts[0].y_pos  # Blinky
        ink_x, ink_y = self.ghosts[1].x_pos, self.ghosts[1].y_pos      # Pinky (đổi thành Inky)
        pink_x, pink_y = self.ghosts[2].x_pos, self.ghosts[2].y_pos    # Inky (đổi thành Pinky)
        clyd_x, clyd_y = self.ghosts[3].x_pos, self.ghosts[3].y_pos    # Clyde

        # Xác định vị trí chạy trốn của ma
        if player_x < 450:
            runaway_x = 900
        else:
            runaway_x = 0
        if player_y < 450:
            runaway_y = 900
        else:
            runaway_y = 0
        
        return_target = (380, 400)  # Điểm trở về hộp của ma

        # Logic mục tiêu khi có powerup
        if self.powerup:
            # Blinky (ghost[0])
            if not self.ghosts[0].dead and not self.eaten_ghost[0]:
                blink_target = (runaway_x, runaway_y)  # Blinky chạy trốn
            elif not self.ghosts[0].dead and self.eaten_ghost[0]:
                if 340 < blink_x < 560 and 340 < blink_y < 500:
                    blink_target = (400, 100)
                else:
                    blink_target = (player_x, player_y)
            else:
                blink_target = return_target

            # Inky (ghost[1], trước đây là Pinky)
            if not self.ghosts[1].dead and not self.eaten_ghost[1]:
                ink_target = (runaway_x, player_y)  # Inky chạy trốn
            elif not self.ghosts[1].dead and self.eaten_ghost[1]:
                if 340 < ink_x < 560 and 340 < ink_y < 500:
                    ink_target = (400, 100)
                else:
                    ink_target = (player_x, player_y)
            else:
                ink_target = return_target

            # Pinky (ghost[2], trước đây là Inky)
            if not self.ghosts[2].dead and not self.eaten_ghost[2]:
                pink_target = (player_x, runaway_y)  # Pinky chạy trốn
            elif not self.ghosts[2].dead and self.eaten_ghost[2]:
                if 340 < pink_x < 560 and 340 < pink_y < 500:
                    pink_target = (400, 100)
                else:
                    pink_target = (player_x, player_y)
            else:
                pink_target = return_target

            # Clyde (ghost[3])
            if not self.ghosts[3].dead and not self.eaten_ghost[3]:
                clyd_target = (450, 450)  # Clyde chạy trốn
            elif not self.ghosts[3].dead and self.eaten_ghost[3]:
                if 340 < clyd_x < 560 and 340 < clyd_y < 500:
                    clyd_target = (400, 100)
                else:
                    clyd_target = (player_x, player_y)
            else:
                clyd_target = return_target
        else:  # Nếu không có powerup
            # Blinky
            if not self.ghosts[0].dead:
                if 340 < blink_x < 560 and 340 < blink_y < 500:
                    blink_target = (400, 100)
                else:
                    blink_target = (player_x, player_y)  # Blinky đuổi Pac-Man
            else:
                blink_target = return_target

            # Inky
            if not self.ghosts[1].dead:
                if 340 < ink_x < 560 and 340 < ink_y < 500:
                    ink_target = (400, 100)
                else:
                    ink_target = (player_x, player_y)  # Inky đuổi Pac-Man
            else:
                ink_target = return_target

            # Pinky
            if not self.ghosts[2].dead:
                if 340 < pink_x < 560 and 340 < pink_y < 500:
                    pink_target = (400, 100)
                else:
                    pink_target = (player_x, player_y)  # Pinky đuổi Pac-Man
            else:
                pink_target = return_target

            # Clyde
            if not self.ghosts[3].dead:
                if 340 < clyd_x < 560 and 340 < clyd_y < 500:
                    clyd_target = (400, 100)
                else:
                    clyd_target = (player_x, player_y)  # Clyde đuổi Pac-Man
            else:
                clyd_target = return_target

        return [blink_target, ink_target, pink_target, clyd_target]
    def reset(self):
        self.powerup = False
        self.power_counter = 0
        self.startup_counter = 0
        self.player = Player()
        self.direction_command = 0
        self.player.score = 0
        self.player.lives = 3
        self.level = copy.deepcopy(boards)
        self.game_over = False
        self.game_won = False
        self.path_to_target = None
        self.current_target_index = 0
        self.game_duration = 0
        self.start_time = None
        # Đặt lại ghosts
        self.eaten_ghost = [False, False, False, False]
        self.startup_counter_ghost = 0
        self.ghosts = [
        Ghost(450, 400, (self.player.x, self.player.y), self.ghost_speeds[0], self.blinky_img, 0, False, True, 0, self.level),
        Ghost(400, 400, (self.player.x, self.player.y), self.ghost_speeds[1], self.pinky_img, 0, False, True, 1, self.level),
        Ghost(500, 400, (self.player.x, self.player.y), self.ghost_speeds[2], self.inky_img, 0, False, True, 2, self.level),
        Ghost(450, 450, (self.player.x, self.player.y), self.ghost_speeds[3], self.clyde_img, 0, False, True, 3, self.level)
        ]
    def run(self):
        run = True
        in_menu = True
        moving = False

        while run:
            self.timer.tick(self.fps)
            if in_menu:
                self.draw_menu()
          
            else:
                # Cập nhật thời gian game nếu chưa thắng
                if self.start_time is None:
                    self.start_time = pygame.time.get_ticks()
                if not self.game_won and not self.game_over:
                    current_time = pygame.time.get_ticks()
                    self.game_duration = (current_time - self.start_time) / 1000.0  # Tính bằng giây
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
                # Ghosts bắt đầu di chuyển sau một khoảng thời gian
                if self.startup_counter_ghost < 60:
                    self.startup_counter_ghost += 1
                else:
                    # Lấy danh sách các mục tiêu cho ghosts
                    targets = self.get_targets()
                    for i, ghost in enumerate(self.ghosts):
                        # Cập nhật mục tiêu của ghost
                        ghost.target = targets[i]
                        
                        ghost.update_state(self.powerup, self.eaten_ghost, self.ghost_speeds[i], ghost.x_pos, ghost.y_pos)
                        if i == 0:
                            if(ghost.in_box or ghost.dead):
                                if(ghost.in_box):
                                    ghost.dead = False
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                            else:
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_blinky()
                        elif i == 1:
                            if(ghost.in_box or ghost.dead):
                                if(ghost.in_box):
                                    ghost.dead = False
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                            else:
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_pinky()
                        elif i == 2:
                            if(ghost.in_box or ghost.dead):
                                if(ghost.in_box):
                                    ghost.dead = False
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                            else:
                                ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_inky()
                        elif i == 3:
                            if(ghost.in_box):
                                    ghost.dead = False
                            ghost.x_pos, ghost.y_pos, ghost.direction = ghost.move_clyde()
                        ghost.center_x = ghost.x_pos + 22
                        ghost.center_y = ghost.y_pos + 22
                        ghost.turns, ghost.in_box = ghost.check_collisions()
                ghost_pos = [
                    self.get_surrounding_positions(self.ghosts[0].center_x, self.ghosts[0].center_y),
                    self.get_surrounding_positions(self.ghosts[1].center_x, self.ghosts[1].center_y),    
                    self.get_surrounding_positions(self.ghosts[2].center_x, self.ghosts[2].center_y),
                    self.get_surrounding_positions(self.ghosts[3].center_x, self.ghosts[3].center_y)
                ]
                self.screen.fill('black')
                self.draw_board()
                center_x, center_y = self.player.get_center()
                
                self.game_won = all(1 not in row and 2 not in row for row in self.level)
                if self.game_won and self.game_mode in (1, 2):  # Lưu thời gian khi thắng
                    if self.game_mode == 1 and self.game_duration not in self.bfs_times:
                        self.bfs_times.insert(0, self.game_duration)  # Thêm vào đầu danh sách
                    elif self.game_mode == 2 and self.game_duration not in self.astar_times:
                        self.astar_times.insert(0, self.game_duration)
                pygame.draw.circle(self.screen, 'black', (center_x, center_y), 20, 2)
                self.player.draw(self.screen, self.counter)
                self.draw_misc()

                self.turns_allowed = self.check_position(center_x, center_y)
                
                if moving and self.game_mode in (1, 2) and not self.game_won:
                    current_grid_pos = self.get_grid_pos(center_x, center_y)
                    
                    if self.path_to_target is None or self.current_target_index >= len(self.path_to_target):
                        dots = self.find_dots()
                        if dots:
                            if self.game_mode == 1:
                                self.path_to_target = self.bfs(current_grid_pos, dots,ghost_pos)
                            else:
                                self.path_to_target = self.a_star(current_grid_pos, dots)
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
                
                elif moving and self.game_mode == 0:
                    self.player.move(self.turns_allowed)
                # Vẽ ghosts
                for ghost in self.ghosts:
                    
                    ghost.draw()
                for ghost in self.ghosts:
                    ghost.target = ()
                self.powerup, self.power_counter = self.check_collisions()
                if self.check_ghost_collisions():  # Gọi hàm mới
                    moving = False  # Dừng di chuyển nếu va chạm nghiêm trọng
                    self.startup_counter = 0  # Đặt lại startup_counter
                if self.player.lives <= 0:
                    self.game_over = True
                    moving = False
                    self.startup_counter = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if in_menu:
                        if event.key == pygame.K_1:
                            self.game_mode = 0
                            in_menu = False
                            self.start_time = pygame.time.get_ticks()  # Bắt đầu đếm thời gian
                        elif event.key == pygame.K_2:
                            self.game_mode = 1
                            in_menu = False
                            self.start_time = pygame.time.get_ticks()  # Bắt đầu đếm thời gian
                        elif event.key == pygame.K_3:
                            self.game_mode = 2
                            in_menu = False
                            self.start_time = pygame.time.get_ticks()  # Bắt đầu đếm thời gian
                    elif self.game_mode == 0:
                        if event.key == pygame.K_RIGHT:
                            self.direction_command = 0
                        elif event.key == pygame.K_LEFT:
                            self.direction_command = 1
                        elif event.key == pygame.K_UP:
                            self.direction_command = 2
                        elif event.key == pygame.K_DOWN:
                            self.direction_command = 3
                    if event.key == pygame.K_SPACE and (self.game_over or self.game_won):
                        self.reset()
                        in_menu = True

                if event.type == pygame.KEYUP and self.game_mode == 0:
                    if event.key == pygame.K_RIGHT and self.direction_command == 0:
                        self.direction_command = self.player.direction
                    elif event.key == pygame.K_LEFT and self.direction_command == 1:
                        self.direction_command = self.player.direction
                    elif event.key == pygame.K_UP and self.direction_command == 2:
                        self.direction_command = self.player.direction
                    elif event.key == pygame.K_DOWN and self.direction_command == 3:
                        self.direction_command = self.player.direction

            if self.game_mode == 0:
                if self.turns_allowed[self.direction_command]:
                    self.player.direction = self.direction_command

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()