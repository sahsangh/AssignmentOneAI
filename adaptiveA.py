import heapq
import numpy as np
import matplotlib.pyplot as plt

h_values = {}  # Global dictionary to store heuristic values

def visualize_step(known_maze, maze, path_taken, current, goal, planned_path, step_count):
    """Visualize the current step of repeated forward A*"""
    plt.figure(figsize=(12, 5))
    
    # Known maze
    plt.subplot(1, 2, 1)
    plt.imshow(known_maze, cmap='gray')
    plt.plot([p[1] for p in path_taken], [p[0] for p in path_taken], 'bo-', linewidth=1, markersize=2)
    plt.plot(current[1], current[0], 'go', markersize=2)  # Current position
    plt.plot(goal[1], goal[0], 'ro', markersize=2)  # Goal position
    plt.title(f'Known Maze - Step {step_count}')
    
    # Real maze
    plt.subplot(1, 2, 2)
    plt.imshow(maze, cmap='gray')
    plt.plot([p[1] for p in path_taken], [p[0] for p in path_taken], 'bo-', linewidth=1, markersize=2)
    plt.plot(current[1], current[0], 'go', markersize=2)  # Current position
    plt.plot(goal[1], goal[0], 'ro', markersize=2)  # Goal position
    if planned_path:
        plt.plot([p[1] for p in planned_path], [p[0] for p in planned_path], 'y--', linewidth=1)
    plt.title(f'Real Maze with Planned Path - Step {step_count}')
    
    plt.tight_layout()
    plt.show()

def visualize_final_path(maze, path_taken, gridNum):
    """Visualize the final path taken by the agent"""
    plt.figure(figsize=(8, 6))
    plt.imshow(maze, cmap='gray')
    plt.plot([p[1] for p in path_taken], [p[0] for p in path_taken], 'bo-', linewidth=2, markersize=2)
    plt.title('Final Path Taken for Grid ' + str(gridNum))
    plt.show()

def adaptive_astar(maze, start, goal, tie_breaking='larger_g', gridNum = None):
    known_maze = np.zeros_like(maze)
    current = start
    path_taken = [current]
    total_expanded_cells = 0
    step_count = 0
    
    def get_neighbors(x, y, maze_state):
        neighbors = []
        possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        
        for nx, ny in possible_moves:
            if 0 <= nx < maze_state.shape[0] and 0 <= ny < maze_state.shape[1] and maze_state[nx, ny] != -1:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def observe_surroundings(real_maze, known_maze, position):
        x, y = position
        for nx, ny in [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < real_maze.shape[0] and 0 <= ny < real_maze.shape[1]:
                known_maze[nx, ny] = real_maze[nx, ny]
    
    def adaptive_heuristic(a, b):
        manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
        h_value = h_values.get(a, manhattan)
        return max(h_value, manhattan)
    
    def astar(start, goal, maze_state):
        if maze_state[start[0], start[1]] == -1:
            return None, 0, {}
        
        max_g = maze_state.shape[0] * maze_state.shape[1]
        c = max_g + 100
        
        open_set = []
        came_from = {}
        g_score = {start: 0}
        f_score = {start: g_score[start] + adaptive_heuristic(start, goal)}
        
        if tie_breaking == 'larger_g':
            priority = (c * f_score[start] - g_score[start], start)
        else: 
            priority = (f_score[start], g_score[start], start)
            
        heapq.heappush(open_set, priority)
        
        closed_set = set()
        expanded_cells = 0
        
        while open_set:
            if tie_breaking == 'larger_g':
                priority_val, current = heapq.heappop(open_set)
            else: 
                _, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            expanded_cells += 1
            
            if current == goal:
                path = reconstruct_path(came_from, current)
                
                goal_g = g_score[goal]
                for node in closed_set:
                    if node in g_score:
                        new_h = goal_g - g_score[node]
                        manhattan = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
                        old_h = h_values.get(node, manhattan)
                        h_values[node] = max(old_h, new_h)
                
                return path, expanded_cells, g_score
            
            for neighbor in get_neighbors(*current, maze_state):
                tentative_g_score = g_score[current] + 1
                
                if neighbor in closed_set:
                    continue
                    
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + adaptive_heuristic(neighbor, goal)
                    
                    if tie_breaking == 'larger_g':
                        priority = (c * f_score[neighbor] - g_score[neighbor], neighbor)
                    else: 
                        priority = (f_score[neighbor], g_score[neighbor], neighbor)
                    
                    heapq.heappush(open_set, priority)
        
        return None, expanded_cells, g_score
    
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    observe_surroundings(maze, known_maze, current)

    while current != goal:
        step_count += 1
        #print(f"\nStep {step_count}: Agent at {current}")
        
        planned_path, expanded, g_scores = astar(current, goal, known_maze)
        total_expanded_cells += expanded
        
        #print(f"Expanded {expanded} cells during planning")
        
        # Visualize the current step
        #visualize_step(known_maze, maze, path_taken, current, goal, planned_path, step_count)
        
        if not planned_path:
            print("No path to goal found with current knowledge!")
            return path_taken, total_expanded_cells
            
        next_pos = planned_path[1] if len(planned_path) > 1 else current
        
        #print(f"Moving from {current} to {next_pos}")
        
        if maze[next_pos[0], next_pos[1]] != -1:
            current = next_pos
            path_taken.append(current)
        else:
            # Update known maze
            known_maze[next_pos[0], next_pos[1]] = -1
        
        observe_surroundings(maze, known_maze, current)
    
    print(f"\nGoal reached in {step_count} steps!")
    print(f"Total expanded cells: {total_expanded_cells}")
    print(f"Final path: {path_taken}")
    
    # Visualize the final path
    #visualize_final_path(maze, path_taken, gridNum)
    
    return path_taken, total_expanded_cells


if __name__ == "__main__":
    expansionValues = []
    for i in range(50):
        h_values.clear()
        filename = f"gridWorlds/gridworld_{i}.npy"
        maze = np.load(filename)
        start = (0, 0)
        goal = (maze.shape[0] - 1, maze.shape[1] - 1)

        path, expanded = adaptive_astar(maze, start, goal, tie_breaking='larger_g', gridNum=i)
        expansionValues.append(expanded)

    print(f"Large: {sum(expansionValues)}")
    print(f"Average Large: {sum(expansionValues)/len(expansionValues)}")


    '''Test running adaptive A* multiple times on one maze to see improvement over time'''
    maze = np.load(f"gridWorlds/multipleAdaptiveA.npy")
    start = (1,7)
    goal = (48,92)
    h_values.clear()
    path, expanded = adaptive_astar(maze, start, goal, tie_breaking='larger_g')
    path, expanded = adaptive_astar(maze, start, goal, tie_breaking='larger_g')
