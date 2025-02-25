import heapq
import numpy as np
import matplotlib.pyplot as plt

def heuristic(a, b, h_values):
    """Returns the heuristic value using updated h-values if available."""
    return h_values.get(a, abs(a[0] - b[0]) + abs(a[1] - b[1]))

def adaptive_astar(maze, start, goal, h_values, tie_breaking='smaller_g'):
    """
    Adaptive A* implementation.
    Uses updated h-values from previous searches to improve efficiency.
    """
    
    def get_neighbors(x, y):
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < maze.shape[0] - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < maze.shape[1] - 1: neighbors.append((x, y + 1))
        return neighbors

    def astar(start, goal):
        """Runs A* search from start to goal."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal, h_values)}
        expanded_cells = 0
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded_cells += 1

            if current == goal:
                return reconstruct_path(came_from, current), expanded_cells, g_score

            closed_set.add(current)

            for neighbor in get_neighbors(*current):
                if neighbor in closed_set or maze[neighbor[0], neighbor[1]] == -1:  # Blocked cell
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, h_values)
                    priority = f_score[neighbor]
                    if tie_breaking == 'larger_g':
                        priority = 1000 * f_score[neighbor] - g_score[neighbor]
                    heapq.heappush(open_set, (priority, neighbor))

        return None, expanded_cells, g_score

    def reconstruct_path(came_from, current):
        """Reconstructs the path from goal to start."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    path, expanded_cells, g_score = astar(start, goal)

    # **Update h-values** for future searches
    if path:
        for s in g_score:
            if s in h_values:
                h_values[s] = max(h_values[s], g_score[goal] - g_score[s])  # Ensure non-decreasing h-values
            else:
                h_values[s] = g_score[goal] - g_score[s]

    return path, expanded_cells, h_values


def compare_tie_breaking(maze, start, goal):
    """Runs Adaptive A* with different tie-breaking strategies and compares performance."""
    h_values = {}  # Store adaptive heuristic values

    path_smaller_g, expanded_smaller_g, h_values = adaptive_astar(maze, start, goal, h_values, tie_breaking='smaller_g')
    path_larger_g, expanded_larger_g, h_values = adaptive_astar(maze, start, goal, h_values, tie_breaking='larger_g')

    print(f"Smaller g-values: Expanded cells = {expanded_smaller_g}")
    print(f"Larger g-values: Expanded cells = {expanded_larger_g}")

    return (path_smaller_g, expanded_smaller_g), (path_larger_g, expanded_larger_g)


def visualize_maze(maze, path, title):
    """Displays the gridworld with the computed path."""
    plt.imshow(maze, cmap='gray')
    if path is not None:
        for (x, y) in path:
            plt.plot(y, x, 'ro', markersize=2)  # Plot path as red dots
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    h_values = {}  # Shared h-values across multiple searches

    for i in range(5,6):
        filename = f"gridWorlds/gridworld_{i}.npy"
        maze = np.load(filename)
        start = (0, 0)
        goal = (maze.shape[0] - 1, maze.shape[1] - 1)

        (smaller_g_result, larger_g_result) = compare_tie_breaking(maze, start, goal)

        visualize_maze(maze, smaller_g_result[0], f'Path with Smaller g-values (gridworld_{i})')
        visualize_maze(maze, larger_g_result[0], f'Path with Larger g-values (gridworld_{i})')
        test_maze = np.array([
        [0, 0, 0, 0, 0, -1],
        [-1, -1, -1, -1, 0, -1],
        [0, 0, 0, -1, 0, -1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, -1],
        [0, 0, 0, 0, 0, 0]
    ])
        x, y,z = adaptive_astar(test_maze, start, (5,5), {},tie_breaking='larger_g')
        visualize_maze(test_maze, x, "Test")