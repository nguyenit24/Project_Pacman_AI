# Dự án Pac-Man AI

Dự án này triển khai trò chơi Pac-Man cổ điển với nhiều thuật toán AI khác nhau để điều khiển Pac-Man. Trò chơi được xây dựng bằng Pygame, và các thuật toán tìm đường và ra quyết định khác nhau được triển khai để điều hướng mê cung, ăn các chấm và tránh ma.

## Tính năng

- **Chơi thủ công:** Điều khiển Pac-Man bằng các phím mũi tên.
- **Thuật toán AI:**
  - Tìm kiếm theo chiều rộng (BFS)
  - Tìm kiếm A\* (A-Star)
  - A\* Thời gian thực (RTA\*)
  - Quay lui (Backtracking)
  - Thuật toán Di truyền (Genetic Algorithm)
  - Học Q Sâu Dựa Trên Mạng Tích Chập (DCQL)
- **Nhiều màn chơi:** Chơi ở các màn chơi khác nhau với số lượng ma khác nhau.
- **Thống kê trò chơi:** Theo dõi và so sánh hiệu suất của các thuật toán AI khác nhau (thời gian, điểm số, mạng sống).
- **AI của Ma:** Các con ma với các kiểu di chuyển riêng biệt (Blinky, Pinky, Inky, Clyde).

## Cách chạy

1.  Đảm bảo bạn đã cài đặt Python và Pygame.
2.  Cài đặt các thư viện phụ thuộc khác:
    ```bash
    pip install numpy torch gymnasium ale-py matplotlib
    ```
3.  Chạy file game chính:
    ```bash
    python game.py
    ```
    Thao tác này sẽ mở cửa sổ trò chơi nơi bạn có thể chọn chế độ chơi (thủ công hoặc thuật toán AI) và màn chơi.

## Cấu trúc file

- `game.py`: Logic chính của trò chơi, vòng lặp Pygame và tích hợp AI.
- `board.py`: Định nghĩa bố cục bảng trò chơi cho các màn chơi khác nhau.
- `player.py`: Triển khai nhân vật người chơi Pac-Man.
- `ghost.py`: Triển khai các nhân vật ma và AI của chúng.
- `logic.py`: Chứa việc triển khai các thuật toán tìm đường và AI (BFS, A\*, RTA\*, Backtracking, Genetic).
- `train_dcqn.py`: Script để huấn luyện mô hình Deep Convolutional Q-Learning.
- `dcql_model_episode_*.pth`: Các file mô hình đã lưu cho tác nhân DCQL.
- `game_stats.csv`: File CSV để lưu trữ số liệu thống kê trò chơi nhằm phân tích hiệu suất.
- `assets/`: Chứa các tài sản của trò chơi như hình ảnh và âm thanh.

## Các thuật toán AI đã triển khai

Các thuật toán AI sau được triển khai trong [`logic.py`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py):

- **BFS:** [`Pathfinder.bfs()`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py)
- **A\*:** [`Pathfinder.A_star()`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py)
- **RTA\*:** [`Pathfinder.rta_star_realtime()`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py)
- **Backtracking:** [`Pathfinder.backtracking()`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py)
- **Genetic Algorithm:** [`Pathfinder.genetic()`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Clogic.py)

Tác nhân Deep Convolutional Q-Learning (DCQL) được huấn luyện bằng cách sử dụng [`train_dcqn.py`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Ctrain_dcqn.py) và được tích hợp vào [`game.py`](d%3A%5Cpy%5CProject_AI_CK%5CProject_Pacman_AI%5Cgame.py).

## Mô tả các Level và Thuật toán AI

Dưới đây là mô tả trực quan về cách các thuật toán AI khác nhau hoạt động trong từng màn chơi.

### Level 1 (Màn chơi cơ bản)

- **BFS (Tìm kiếm theo chiều rộng):**

  ![BFS Level 1 GIF](assets/gifs/Level1_gif/1.gif)

- **A\* (A-Star):**

  ![A* Level 1 GIF](assets/gifs/Level1_gif/2.gif)

- **RTA\* (A\* Thời gian thực):**

  ![RTA* Level 1 GIF](assets/gifs/Level1_gif/3.gif)

- **Backtracking (Quay lui):**

  ![Backtracking Level 1 GIF](assets/gifs/Level1_gif/4.gif)

- **Genetic Algorithm (Thuật toán Di truyền):**

  ![Genetic Level 1 GIF](assets/gifs/Level1_gif/5.gif)

- **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập):**

  ![DCQL Level 1 GIF](assets/gifs/Level1_gif/6.gif)

|        **BFS (Tìm kiếm theo chiều rộng)**        |                **A\* (A-Star)**                 |          **RTA\* (A\* Thời gian thực)**           |
| :----------------------------------------------: | :---------------------------------------------: | :-----------------------------------------------: |
| ![BFS Level 1 GIF](assets/gifs/Level1_gif/1.gif) | ![A* Level 1 GIF](assets/gifs/Level1_gif/2.gif) | ![RTA* Level 1 GIF](assets/gifs/Level1_gif/3.gif) |

|                **Backtracking (Quay lui)**                |     **Genetic Algorithm (Thuật toán Di truyền)**     |   **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập)**    |
| :-------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------: |
| ![Backtracking Level 1 GIF](assets/gifs/Level1_gif/4.gif) | ![Genetic Level 1 GIF](assets/gifs/Level1_gif/5.gif) | ![DCQL Level 1 GIF](assets/gifs/Level1_gif/6.gif) |

#### Thống kê Level 1

![Thống kê Level 1](assets/stats/level1_stats.jpg)

### Level 2 (Màn chơi với nhiều ma hơn)

- **BFS (Tìm kiếm theo chiều rộng):**

  ![BFS Level 2 GIF](assets/gifs/level2_gif/1.gif)

- **A\* (A-Star):**

  ![A* Level 2 GIF](assets/gifs/level2_gif/2.gif)

- **RTA\* (A\* Thời gian thực):**

  ![RTA* Level 2 GIF](assets/gifs/level2_gif/3.gif)

- **Backtracking (Quay lui):**

  ![Backtracking Level 2 GIF](assets/gifs/level2_gif/4.gif)

- **Genetic Algorithm (Thuật toán Di truyền):**

  ![Genetic Level 2 GIF](assets/gifs/level2_gif/5.gif)

- **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập):**

  ![DCQL Level 2 GIF](assets/gifs/level2_gif/6.gif)

|        **BFS (Tìm kiếm theo chiều rộng)**        |                **A\* (A-Star)**                 |          **RTA\* (A\* Thời gian thực)**           |
| :----------------------------------------------: | :---------------------------------------------: | :-----------------------------------------------: |
| ![BFS Level 2 GIF](assets/gifs/level2_gif/1.gif) | ![A* Level 2 GIF](assets/gifs/level2_gif/2.gif) | ![RTA* Level 2 GIF](assets/gifs/level2_gif/3.gif) |

|                **Backtracking (Quay lui)**                |     **Genetic Algorithm (Thuật toán Di truyền)**     |   **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập)**    |
| :-------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------: |
| ![Backtracking Level 2 GIF](assets/gifs/level2_gif/4.gif) | ![Genetic Level 2 GIF](assets/gifs/level2_gif/5.gif) | ![DCQL Level 2 GIF](assets/gifs/level2_gif/6.gif) |

#### Thống kê Level 2

![Thống kê Level 2](assets/stats/level2_stats.jpg)

### Level 3 (Màn chơi phức tạp)

- **BFS (Tìm kiếm theo chiều rộng):**

  ![BFS Level 3 GIF](assets/gifs/level3_gif/1.gif)

- **A\* (A-Star):**

  ![A* Level 3 GIF](assets/gifs/level3_gif/2.gif)

- **RTA\* (A\* Thời gian thực):**

  ![RTA* Level 3 GIF](assets/gifs/level3_gif/3.gif)

- **Backtracking (Quay lui):**

  ![Backtracking Level 3 GIF](assets/gifs/level3_gif/4.gif)

- **Genetic Algorithm (Thuật toán Di truyền):**

  ![Genetic Level 3 GIF](assets/gifs/level3_gif/5.gif)

- **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập):**

  ![DCQL Level 3 GIF](assets/gifs/level3_gif/6.gif)

|        **BFS (Tìm kiếm theo chiều rộng)**        |                **A\* (A-Star)**                 |          **RTA\* (A\* Thời gian thực)**           |
| :----------------------------------------------: | :---------------------------------------------: | :-----------------------------------------------: |
| ![BFS Level 3 GIF](assets/gifs/level3_gif/1.gif) | ![A* Level 3 GIF](assets/gifs/level3_gif/2.gif) | ![RTA* Level 3 GIF](assets/gifs/level3_gif/3.gif) |

|                **Backtracking (Quay lui)**                |     **Genetic Algorithm (Thuật toán Di truyền)**     |   **DCQL (Học Q Sâu Dựa Trên Mạng Tích Chập)**    |
| :-------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------: |
| ![Backtracking Level 3 GIF](assets/gifs/level3_gif/4.gif) | ![Genetic Level 3 GIF](assets/gifs/level3_gif/5.gif) | ![DCQL Level 3 GIF](assets/gifs/level3_gif/6.gif) |

#### Thống kê Level 3

![Thống kê Level 3](assets/stats/level3_stats.jpg)

- Ở môi trường level 3, có 2 ghost di chuyển liên tục, tăng độ khó và phức tạp của môi trường động và đa mục tiêu (vừa ăn pellet vừa tránh ghost). Vì vậy có rất nhiều agent không thể hoàn thiện: 
+ A*: A* vẫn đem lại hiệu suất tốt nhất với thời gian chạy nhanh nhất và khả năng tránh ghost tốt để không bị mất mạng nào. Giải thuật A* tính toán toàn cục kết hợp với việc tối ưu hàm heuristic, phản ứng thời gian thực và tái tính toán đường đi, cùng với đó là việc thêm penalty phạt nặng khi gần ghost đã mang lại hiệu quả tương đối ổn định cho cả hai mục tiêu ăn pellet và tránh ghost. Các yếu tố trên vẫn đáp ứng được trong môi trường hai ghost. 
+ RTA*: RTA* cũng hoàn thiện trong màn chơi này. Thuật toán thể hiện được khả năng phản ứng nhanh trước các thay đổi của môi trường (ở đây là vị trí ghost thay đổi liên tục) từ đó có khả năng tránh ghost hoàn hảo. Tuy nhiên điểm yếu về không tối ưu toàn cục tốt đã làm cho agent mất nhiều thời gian hơn để hoàn thành mục tiêu ăn hết viên pellet. 
+ Genetic: GA không hoàn thiện trong màn chơi này. Kết quả trên chứng tỏ hàm fitness đã bộc lộ điểm yếu trong việc thực hiện cả hai mục tiêu ăn pellet và tránh ghost. GA hoạt động theo thế hệ, cần nhiều thời gian để tính toán quần thể, lai ghép, đột biến, và đánh giá fitness. Trong môi trường động, vị trí của ma thay đổi liên tục, nhưng GA không thể cập nhật đường đi theo thời gian thực vì nó dựa trên một quần thể tĩnh tại mỗi lần chạy. 
+ BFS, Backtracking: Vẫn có số điểm khá cao nhưng lại không hoàn thiện trong màn chơi. Agent mất hết số mạng do va chạm với ghost. Điều đó thể hiện đúng bản chất của BFS là thuật toán tìm kiếm mù, vì vậy không thể xác định được vị trí của ghost để né. Trong khi backtracking thì cũng không phải là thuật toán tối ưu để xử lý thay đổi thời gian thực và liên tục, ở đây là vị trí của ghost.
+ DCQL: agent chưa được train đủ lâu và bộ dữ liệu chưa nhiều nên kết quả còn khá khiêm tốn.


