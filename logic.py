import heapq
from collections import deque
import random


class Pathfinder:
    def __init__(self, level):
        self.level = level

    def bfs(self, start, target_list, ghost_pos):
        if not target_list:
            return None
        queue = deque([(start, [])])
        visited = set([start])

        # Làm phẳng ghost_pos thành tập hợp các vị trí duy nhất
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
                # Kiểm tra điều kiện: trong lưới, không phải tường, chưa thăm,
                # và không nằm trong ghost_positions
                if 0 <= new_row < len(self.level) and 0 <= new_col < len(self.level[0]) and self.level[new_row][new_col] < 3 and (new_row, new_col) not in visited and (new_row, new_col) not in ghost_positions:
                    queue.append(((new_row, new_col), path + [(row, col)]))
                    visited.add((new_row, new_col))
        return None

    def is_valid_position(self, pos):
        """Kiểm tra trong biên, không phải tường (level < 3)."""
        row, col = pos
        return (
            0 <= row < len(self.level) and
            0 <= col < len(self.level[0]) and
            self.level[row][col] < 3
        )

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

                if 0 <= new_row < len(self.level) and 0 <= new_col < len(self.level[0]) and self.level[new_row][new_col] < 3 and new_pos not in visited:

                    tentative_g_score = g_scores[current] + 1
                    if new_pos not in g_scores or tentative_g_score < g_scores[new_pos]:
                        g_scores[new_pos] = tentative_g_score
                        h_score = min(self.heuristic(new_pos, target) for target in target_list)
                        f_score_new = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score_new, new_pos, path + [current]))
        return None

   

        

    def A_star(self, start, target_list, ghost_positions=None, ghost_radius = 4):
 
        if not target_list:
            return None

        flat_ghosts = []
        if ghost_positions:
            for group in ghost_positions:
                if isinstance(group, list):
                    flat_ghosts.extend(group)
                else:
                    flat_ghosts.append(group)

        very_close_threshold = ghost_radius * 0.7

        def compute_penalty(pos, ghosts, ghost_radius):
            penalty = 0
            for ghost in ghosts:
                d = self.heuristic(pos, ghost)
                if d < very_close_threshold:
                    penalty += 10000  # Tránh tuyệt đối
                elif d < ghost_radius:
                    penalty += (ghost_radius - d) * 50  # Tăng penalty để ưu tiên xa ghost
            return penalty

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

                if (0 <= new_row < len(self.level) and 
                    0 <= new_col < len(self.level[0]) and 
                    self.level[new_row][new_col] < 3 and 
                    new_pos not in visited):
                    penalty = compute_penalty(new_pos, flat_ghosts, ghost_radius) if flat_ghosts else 0
                    tentative_g_score = g_scores[current] + 1 + penalty
                    if new_pos not in g_scores or tentative_g_score < g_scores[new_pos]:
                        g_scores[new_pos] = tentative_g_score
                        h_score = min(self.heuristic(new_pos, target) for target in target_list)
                        f_score_new = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score_new, new_pos, path + [current]))
        return None
    def rta_star_avoid_ghosts(self, start, target_list, ghost_positions=None, ghost_radius=3, depth_limit=5):
        if not target_list:
            return None

        # Làm phẳng ghost_positions
        flat_ghosts = []
        if ghost_positions:
            for group in ghost_positions:
                if isinstance(group, list):
                    flat_ghosts.extend(group)
                else:
                    flat_ghosts.append(group)

        very_close_threshold = ghost_radius * 0.7

        def compute_penalty(pos, ghosts, ghost_radius):
            penalty = 0
            for ghost in ghosts:
                d = self.heuristic(pos, ghost)
                if d < very_close_threshold:
                    penalty += 10000  # Tránh tuyệt đối
                elif d < ghost_radius:
                    penalty += (ghost_radius - d) * 50  # Tăng penalty để ưu tiên xa ghost
            return penalty

        # Mỗi nút: (f_score, current, path, depth)
        open_set = [(0, start, [], 0)]
        heapq.heapify(open_set)
        visited = set()
        best_candidate = None
        best_f = float("inf")

        while open_set:
            f, current, path, depth = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            # Nếu đạt độ sâu giới hạn, cập nhật bước đi hiện tại nếu tốt hơn
            if depth >= depth_limit:
                if f < best_f:
                    best_candidate = path + [current]
                    best_f = f
                continue

            # Nếu current là mục tiêu thì trả về
            if current in target_list:
                return path + [current]

            # Mở rộng nút
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_r, new_c = current[0] + dx, current[1] + dy
                new_pos = (new_r, new_c)
                if (0 <= new_r < len(self.level) and 
                    0 <= new_c < len(self.level[0]) and 
                    self.level[new_r][new_c] < 3 and 
                    new_pos not in visited):
                    penalty = compute_penalty(new_pos, flat_ghosts, ghost_radius) if flat_ghosts else 0
                    g_new = (len(path) + 1) + penalty
                    h = min(self.heuristic(new_pos, targ) for targ in target_list)
                    heapq.heappush(open_set, (g_new + h, new_pos, path + [current], depth + 1))

        if best_candidate and len(best_candidate) >= 2:
            return best_candidate[:2]  # Trả về [start, next_step]
        elif best_candidate:
            return best_candidate  # Trả về bước duy nhất nếu chỉ có một bước
        else:
            return None
    def find_dot_safe(self, current_pos, ghost_pos, ghost_radius = 4):
        """Tìm dot an toàn và điểm an toàn nếu cần."""
        dots = []
        for i in range(len(self.level)):
            for j in range(len(self.level[0])):
                if self.level[i][j] in (1, 2):
                    dots.append((i, j))
        if not dots:
            return None, None

        # Sắp xếp dots theo khoảng cách đến ghost gần nhất (ưu tiên dot xa ghost)
        dots_sorted = sorted(dots, key=lambda dot: min(self.heuristic(dot, ghost) for ghost in ghost_pos), reverse=True)
        
        safe_dot = None
        safe_point = None
        for dot in dots_sorted:
            path = self.rta_star_avoid_ghosts(current_pos, [dot], ghost_positions=ghost_pos)
            if path and self.is_path_safe(path, ghost_pos, ghost_radius):
                safe_dot = dot
                break

        # Nếu không có dot an toàn, tìm điểm an toàn
        if not safe_dot:
            for i in range(len(self.level)):
                for j in range(len(self.level[0])):
                    if self.level[i][j] < 3:
                        pos = (i, j)
                        if all(self.heuristic(pos, ghost) >= 5 for ghost in ghost_pos):
                            path = self.rta_star_avoid_ghosts(current_pos, [pos], ghost_positions=ghost_pos)
                            if path:
                                safe_point = pos
                                break
                if safe_point:
                    break

        return safe_dot, safe_point

    def backtracking(self, start, dots, ghost_positions=None, depth_limit=50):
        """
        Thuật toán Backtracking để tìm đường từ 'start' đến một trong các 'dots'.
        Né tránh các vị trí trong 'ghost_positions'.
        """
        if start in dots:
            return [start]

        # Vị trí ghost cần tránh
        avoid = set()
        if ghost_positions:
            for area in ghost_positions:
                avoid.update(area)

        visited = set()
        best_path = None

        def is_safe(pos):
            return pos not in avoid and self.is_valid_position(pos)

        def dfs(current, path, depth):
            nonlocal best_path

            if depth > depth_limit:
                return
            if current in visited or not is_safe(current):
                return
            if best_path and len(path) >= len(best_path):
                return

            visited.add(current)
            path.append(current)

            if current in dots:
                if not best_path or len(path) < len(best_path):
                    best_path = path.copy()
                visited.remove(current)
                path.pop()
                return

            # Chọn hướng theo heuristic
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)
                if is_safe(next_pos):
                    h = min(self.heuristic(next_pos, dot) for dot in dots)
                    neighbors.append((h, next_pos))

            neighbors.sort()  # Ưu tiên heuristic nhỏ hơn

            for _, next_pos in neighbors:
                dfs(next_pos, path, depth + 1)

            visited.remove(current)
            path.pop()

        dfs(start, [], 0)
        return best_path

    def genetic(self, start, goals, ghost_positions=None, max_generations=200, population_size=60):
        if not goals:
            return None

        avoid = set(ghost_positions) if ghost_positions else set()

        def is_safe(pos):
            return pos not in avoid and self.is_valid_position(pos)

        def generate_individual():
            max_length = 20  # Giảm số bước tối đa để tăng tốc
            return [random.choice([(0, 1), (0, -1), (-1, 0), (1, 0)]) for _ in range(max_length)]

        def is_opposite_move(prev_move, curr_move):
            return (prev_move[0] + curr_move[0] == 0 and prev_move[1] + curr_move[1] == 0)

        def fitness(individual):
            pos = start
            score = 0
            visited = set([pos])
            very_close_threshold = 2.8
            ghost_radius = 4
            prev_move = None
            max_steps = 15  # Giới hạn số bước kiểm tra để tăng tốc

            for i, move in enumerate(individual[:max_steps]):
                if prev_move and is_opposite_move(prev_move, move):
                    score -= 2000  # Phạt nếu quay đầu
                    break
                next_pos = (pos[0] + move[0], pos[1] + move[1])
                if not is_safe(next_pos) or next_pos in visited:
                    score -= 2000  # Phạt nếu vị trí không hợp lệ hoặc lặp
                    break
                visited.add(next_pos)
                pos = next_pos
                score += 30  # Thưởng cho mỗi bước hợp lệ
                prev_move = move
                # Phạt nếu gần ghost
                if avoid:
                    min_ghost_dist = min(self.heuristic(pos, ghost) for ghost in avoid)
                    if min_ghost_dist < very_close_threshold:
                        score -= 20000  # Phạt nặng nếu quá gần
                        break
                    elif min_ghost_dist < ghost_radius:
                        score -= (ghost_radius - min_ghost_dist) * 1000  # Phạt nếu gần

            # Thưởng lớn nếu đến mục tiêu, phạt nhẹ nếu không đến
            if pos in goals:
                score += 30000  # Thưởng lớn nếu đến dot
            else:
                score -= 3000  # Phạt nhẹ nếu không đến dot
                min_dist = min(self.heuristic(pos, goal) for goal in goals)
                score += max(0, 150 - min_dist * 15)  # Thưởng dựa trên khoảng cách

            return score if score > -10000 else -10000  # Điểm tối thiểu cho cá thể không khả thi

        def mutate(individual):
            mutation_rate = 0.2
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[i] = random.choice([(0, 1), (0, -1), (-1, 0), (1, 0)])

        def crossover(parent1, parent2):
            if len(parent1) < 2 or len(parent2) < 2:
                return parent1.copy()
            split1 = random.randint(1, len(parent1) - 2)
            split2 = random.randint(split1, len(parent1) - 1)
            return parent1[:split1] + parent2[split1:split2] + parent1[split2:]

        population = [generate_individual() for _ in range(population_size)]
        best_path = None
        best_fitness = float('-inf')
        stagnation_counter = 0
        stagnation_limit = 20

        for generation in range(max_generations):
            population = sorted(population, key=fitness, reverse=True)
            current_best_fitness = fitness(population[0])

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit:
                break

            best_individual = population[0]
            path = [start]
            pos = start
            visited = set([pos])
            prev_move = None
            for move in best_individual:
                if prev_move and is_opposite_move(prev_move, move):
                    break
                next_pos = (pos[0] + move[0], pos[1] + move[1])
                if not is_safe(next_pos) or next_pos in visited:
                    break
                path.append(next_pos)
                visited.add(next_pos)
                pos = next_pos
                prev_move = move
                if pos in goals:
                    break

            # Chấp nhận path nếu nó hợp lệ
            if path and len(path) > 1:
                best_path = path

            elite_size = population_size // 2
            new_population = population[:elite_size]
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population[:elite_size], 2)
                child = crossover(parent1, parent2)
                mutate(child)
                new_population.append(child)

            population = new_population

        return best_path if best_path else None