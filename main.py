import random
import numpy as np
import matplotlib.pyplot as plt

def generate_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
    stack = []
    visited = set()

    def get_neighbors(x, y):
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < width - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < height - 1: neighbors.append((x, y + 1))
        return neighbors

    def dfs(x, y):
        stack.append((x, y))
        visited.add((x, y))
        maze[y, x] = 1  # Mark as unblocked

        while stack:
            current = stack[-1]
            neighbors = [n for n in get_neighbors(*current) if n not in visited]
            if not neighbors:
                stack.pop()
                continue

            next_cell = random.choice(neighbors)
            if random.random() < 0.3:
                maze[next_cell[1], next_cell[0]] = -1  # Mark as blocked
            else:
                maze[next_cell[1], next_cell[0]] = 1  # Mark as unblocked
                stack.append(next_cell)
            visited.add(next_cell)

    # Start DFS from a random cell
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
    dfs(start_x, start_y)

    # Ensure all cells are visited
    for y in range(height):
        for x in range(width):
            if (x, y) not in visited:
                dfs(x, y)

    return maze

def save_mazes(num_mazes, width, height, filename_prefix):
    for i in range(num_mazes):
        maze = generate_maze(width, height)
        np.save(f"{filename_prefix}_{i}.npy", maze)

def load_maze(filename):
    return np.load(filename)

def visualize_maze(maze):
    plt.imshow(maze, cmap='magma')
    plt.show()

def generate_test_map(width, height):
    maze = np.ones((height, width), dtype=int)  # Start with all cells unblocked

    # Block the three surrounding squares to the bottom right corner
    if width > 1 and height > 1:
        maze[height - 1, width - 2] = -1  # Block left of bottom right corner
        maze[height - 2, width - 1] = -1  # Block above bottom right corner
        maze[height - 2, width - 2] = -1  # Block diagonal to bottom right corner
    return maze




if __name__ == "__main__":
    num_mazes = 50
    width, height = 101, 101
    filename_prefix = "gridWorlds/gridworld"

    save_mazes(num_mazes, width, height, filename_prefix)

    # Example of loading and visualizing a maze
    # for i in range(50):
    #     maze = load_maze(f"{filename_prefix}_{i}.npy")
    #     visualize_maze(maze)
    # maze = load_maze(f"{filename_prefix}_0.npy")
    # visualize_maze(maze)
    # blocked = generate_test_map(101, 101)
    # np.save("test_maze.npy", blocked)
    # visualize_maze(blocked)