import heapq
from collections import deque


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

    def rta_star_avoid_ghosts(self, start, target_list, ghost_positions=None, ghost_radius=3, depth_limit=5):
        """
        Phiên bản Real-Time A* (RTA*) cho tránh ghost:
        - Tìm kiếm chỉ đến độ sâu depth_limit.
        - Trả về bước đi đầu tiên (hoặc đường ngắn giới hạn) với chi phí thấp nhất,
        ưu tiên tránh ghost theo penalty.
        """
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
                    penalty += 10000  # cực kỳ tránh nước đi này
                elif d < ghost_radius:
                    penalty += (ghost_radius - d) * 33
            return penalty

        # Mỗi nút: (f_score, current, path, depth)
        open_set = [(0, start, [], 0)]
        heapq.heapify(open_set)
        best_candidate = None
        best_f = float("inf")

        while open_set:
            f, current, path, depth = heapq.heappop(open_set)
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
                if 0 <= new_r < len(self.level) and 0 <= new_c < len(self.level[0]) and self.level[new_r][new_c] < 3:
                    base = 1
                    pen = compute_penalty(new_pos, flat_ghosts, ghost_radius) if flat_ghosts else 0
                    g_new = (len(path) + 1) + pen
                    h = min(self.heuristic(new_pos, targ) for targ in target_list)
                    heapq.heappush(open_set, (g_new + h, new_pos, path + [current], depth + 1))

        # Nếu không đạt được mục tiêu trong giới hạn, chọn bước đi tốt nhất ở mức depth_limit
        if best_candidate and len(best_candidate) >= 2:
            return best_candidate[:2]  # Trả về [start, next_step]
        else:
            return None
