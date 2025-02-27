import heapq
import numpy as np
import matplotlib.pyplot as plt

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def repeated_backward_astar(maze, start, goal, tie_breaking='smaller_g'):
    def get_neighbors(x, y):
        neighbors = []
        if x > 0: neighbors.append((x - 1, y))
        if x < maze.shape[0] - 1: neighbors.append((x + 1, y))
        if y > 0: neighbors.append((x, y - 1))
        if y < maze.shape[1] - 1: neighbors.append((x, y + 1))
        return neighbors

    def astar(start, goal):
        if maze[goal[0], goal[1]] == -1:
            return [], 0
        open_set = []
        heapq.heappush(open_set, (0, goal))
        came_from = {}
        g_score = {goal: 0}
        f_score = {goal: heuristic(goal, start)}
        expanded_cells = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded_cells += 1

            if current == start:
                return reconstruct_path(came_from, current), expanded_cells

            for neighbor in get_neighbors(*current):
                if maze[neighbor[0], neighbor[1]] == -1:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, start)
                    priority = f_score[neighbor]
                    if tie_breaking == 'larger_g':
                        priority = 1000 * f_score[neighbor] - g_score[neighbor]
                    heapq.heappush(open_set, (priority, neighbor))

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

def compare_tie_breaking(maze, start, goal):
    path_smaller_g, expanded_smaller_g = repeated_backward_astar(maze, start, goal, tie_breaking='smaller_g')
    path_larger_g, expanded_larger_g = repeated_backward_astar(maze, start, goal, tie_breaking='larger_g')

    print(f"Smaller g-values: Expanded cells = {expanded_smaller_g}")
    print(f"Larger g-values: Expanded cells = {expanded_larger_g}")

    return (path_smaller_g, expanded_smaller_g), (path_larger_g, expanded_larger_g)

def visualize_maze(maze, path, title):
    plt.imshow(maze, cmap='gray')
    if path is not None:
        for (x, y) in path:
            plt.plot(y, x, 'ro', markersize=2)  # Plot path as red dots
    plt.title(title)
    plt.show()

def analyzeBackward(expansionValues):
    print(expansionValues)
    print(f"{len(expansionValues)} {sum(expansionValues)}")
    total_sum, count = 0, 0
    for i in range(len(expansionValues)):
        if expansionValues[i] != 0:
            total_sum += expansionValues[i]
            count += 1
    print(f"Average expansion: {total_sum/count} and count: {count}")

if __name__ == "__main__":
    expansionValues = []
    for i in range(50):
        filename = f"gridWorlds/gridworld_{i}.npy"
        maze = np.load(filename)
        start = (0, 0)
        goal = (maze.shape[0] - 1, maze.shape[1] - 1)

        '''Larger vs Smaller Test'''
        # (smaller_g_result, larger_g_result) = compare_tie_breaking(maze, start, goal)
        # visualize_maze(maze, smaller_g_result[0], f'Path with Smaller g-values (gridworld_{i})')
        # visualize_maze(maze, larger_g_result[0], f'Path with Larger g-values (gridworld_{i})')

        '''
        TEST FOR FORWARD VS BACKWARD ASTAR
        ONLY USES LARGER G VALUES FOR TIE BREAKS
        '''
        path_larger_g, expanded_larger_g = repeated_backward_astar(maze, start, goal, tie_breaking='larger_g')
        #visualize_maze(maze, path_larger_g, f'Path with Larger g-values (gridworld_{i})')
        print(f"Expanded cells: {expanded_larger_g}")
        expansionValues.append(expanded_larger_g)
    analyzeBackward(expansionValues)