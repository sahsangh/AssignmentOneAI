import heapq
import numpy as np
import matplotlib.pyplot as plt
import time

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def repeated_forward_astar(maze, start, goal, tie_breaking='smaller_g'):
    def get_neighbors(x, y):
        neighbors = []
        possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for nx, ny in possible_moves:
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] != -1:
                neighbors.append((nx, ny))
        
        return neighbors

    def astar(start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        expanded_cells = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded_cells += 1

            print(f"Expanding node: {current}, f: {f_score[current]}, g: {g_score[current]}")

            if current == goal:
                return reconstruct_path(came_from, current), expanded_cells

            for neighbor in get_neighbors(*current):
                if maze[neighbor[0], neighbor[1]] == -1:  # Check if the cell is not blocked
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    priority = f_score[neighbor]
                    if tie_breaking == 'larger_g':
                        priority = 1000 * f_score[neighbor] - g_score[neighbor]
                    heapq.heappush(open_set, (priority, neighbor))
                    print(f"  Adding neighbor: {neighbor}, f: {f_score[neighbor]}, g: {g_score[neighbor]}")

            # Visualize the current state of the maze and path
            current_path = reconstruct_path(came_from, current)
            visualize_maze(maze, current_path, f'Step {expanded_cells}')
            time.sleep(0.5)  # Pause to visualize each step

        return None, expanded_cells

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    path, expanded_cells = astar(start, goal)
    return path, expanded_cells

def visualize_maze(maze, path, title):
    plt.imshow(maze, cmap='gray')
    if path is not None:
        for (x, y) in path:
            plt.plot(y, x, 'ro', markersize=2)  # Plot path as red dots
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Small test case
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0,0, -1, 0, 0],
        [0,0, -1, -1, 0],
        [0,0, -1, -1, 0],
        [0,0, 0, -1, 0]
    ])
    start = (4, 1)
    goal = (4,4)

    path, expanded_cells = repeated_forward_astar(maze, start, goal, tie_breaking='larger_g')
    print(f"Path: {path}")
    print(f"Expanded cells: {expanded_cells}")

    visualize_maze(maze, path, 'Final Path')