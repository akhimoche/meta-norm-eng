# Section 0: Standard library imports
import heapq

# Section 1: Manhattan distance function 
def manhattan_distance(pos1, pos2):
    row1, col1 = pos1
    row2, col2 = pos2
    return abs(row1 - row2) + abs(col1 - col2)

# Section 2: A* algorithm
def a_star(start, goal, obstacles):
    if start == goal:
        return [start]
    
    # Calculate grid dimensions from all positions (start, goal, obstacles)
    # This ensures A* knows the boundaries without requiring external grid_size parameter
    all_positions = [start, goal] + list(obstacles)
    max_row = max(pos[0] for pos in all_positions)
    max_col = max(pos[1] for pos in all_positions)
    rows, cols = max_row + 1, max_col + 1
    # Set up data structures
    open_set = []  # Priority queue: (f_score, position) - positions to explore
    came_from = {}  # Path reconstruction
    g_score = {start: 0}  # Actual costs: g_score[position] = cost_from_start
    
    # Add starting position to exploration queue
    # f_score = g_score + h_score (actual cost + estimated cost to goal)
    h_start = manhattan_distance(start, goal)  # Heuristic: estimated cost to goal
    f_start = g_score[start] + h_start  # Total estimated cost
    heapq.heappush(open_set, (f_start, start))
    
    # Main exploration loop: keep exploring until we find goal or run out of options
    while open_set:
        # Pop most promising position (lowest f_score = most promising)
        current_f_score, current = heapq.heappop(open_set)
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path by following breadcrumbs backwards
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()  # Convert backwards path to forwards path

            return path
        
        # Explore all 4 neighbors (up, right, down, left)
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Skip invalid neighbors
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue  # Out of bounds
            if neighbor in obstacles:
                continue  # Blocked by obstacle
            
            # Calculate tentative cost to reach this neighbor through current path
            tentative_g_score = g_score[current] + 1  # Each step costs 1
            
            # Check if this is a better path to the neighbor
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # Record this better path
                came_from[neighbor] = current  # "I got to neighbor from current"
                g_score[neighbor] = tentative_g_score  # Update best cost to reach neighbor
                
                # Calculate f_score for neighbor and add to exploration queue
                h_neighbor = manhattan_distance(neighbor, goal)  # Heuristic cost
                f_neighbor = tentative_g_score + h_neighbor  # Total estimated cost
                heapq.heappush(open_set, (f_neighbor, neighbor))
    
    # No path found - explored all possible positions
    return []