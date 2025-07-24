#utility functions for spatial reasoning and action planning. 
# These are useful if you're trying to guide agents with logic rather than learning alone (e.g. handing out conventions or routes from an LLM planner)
# Heuristics here were used as a baseline to compare with LLM policy planning? 
import heapq
import numpy as np

def a_star(start: tuple, goal: tuple, obstacles: dict, grid_size: tuple):
    """Perform A* search for shortest path between start and goal indices given
    some obstacles to avoid.
    """
    # Define our heuristic h(x)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # To find neighbouring nodes
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional movement

    # To track nodes yet to be explored
    open_set = []
    # Add initial coords with f(x)=0 (placeholder)
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # Loop through neighbours until goal state reached
    while open_set: # While open_set is non-empty...
        _, current = heapq.heappop(open_set) # ...get the node with smallest f(x)

        # Check goal state has been reached...
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        # ... Otherwise, look through neighbours
        for dx, dy in neighbors: # look in each direction
            neighbor = (current[0] + dx, current[1] + dy)
            # if neighbour is not an obstacle
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1] and neighbor not in obstacles:
                # g(x) cost to move to neighbour
                tentative_g_score = g_score[current] + 1
                # check that neighbour hasn't already been visited, inf guarantees that it will be added
                # otherwise, if revisiting, checks that the current g_score is less than old path one.
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current # record the parent node for retracing the path
                    g_score[neighbor] = tentative_g_score # record g_score for this node
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal) # f(x)=g(x)+h(X)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def get_north(orientation: str):
    """Rotates an agent to north given initial orientation"""

    if orientation == "north":
        action = 0 #"noop"
        new_orientation = "north"

    if orientation == "east":
        action = 5 #"turn_left"
        new_orientation = "north"

    if orientation == "south":
        action = 5 #"turn_left"
        new_orientation = "east"

    if orientation == "west":
        action = 6 #"turn_right"
        new_orientation = "north"

    return action, new_orientation

def move_agent(coord_init: tuple, coord_final: tuple):
    """ Given two coordinates, select the appropriate action
        to move from coord_init to coord_final. All actions
        are relative to north (call get_north before using)."""

    # If same row but coord_final is to the left of coord_init...
    if coord_init[0] == coord_final[0] and coord_init[1] > coord_final[1]:
        move_action = "step_left"

    # If same row but coord_final is to the right of coord_init...
    if coord_init[0] == coord_final[0] and coord_init[1] < coord_final[1]:
        move_action = "step_right"

    # If same col but coord_final is below coord_init...
    if coord_init[0] < coord_final[0] and coord_init[1] == coord_final[1]:
        move_action = "backward"

    # If same col but coord_final is above coord_init...
    if coord_init[0] > coord_final[0] and coord_init[1] == coord_final[1]:
        move_action = "forwards"

    return move_action

def get_movement_actions(operator_output, colour_dict, num_players):
    """ Gets movement actions for each agent by calling
    A* search on the tuple designated by the operator_output.
    operator_output is a dictionary with a key each agent colour
    that maps to a tuple (initial, final, obstacles)
    of coordinates which are fed into A*. The first leg of the
    located path is used. operator_output should be in dictionary form.
    """
    # to be used after a planner (like an LLM) has output an operator_output dictionary saying smth like:
    # "Move the red agent from (2,2) to (4,5), avoiding obstacles at [(3,3), (4,3)]"
    # This is how we translate commands form LLM into actions, but:
    # 1. Not used in mp_testbed, ig since it's meant to be a baseline? 2. No refusal mechanism for agent? 
    move_actions = np.zeros((num_players),dtype=int)

    # do A* search for each agent using operator_output
    for i in range(num_players):
        agent_colour = colour_dict[i] # get colour from index
        search_tuple = operator_output[agent_colour] # (init, fin, obst)
        path = a_star(search_tuple)  # get path from init to fin
        coords_i = path[0], path[1] # we only want the first move
        move_i = move_agent(path[0], path[1]) #

    return move_actions


# Test algorithm
# start = (0, 0)
# goal = (5, 5)
# obstacles = {(1,1),(0,1),(2,2),(2,4),(2,5)}  # Blocked positions on the grid
# path = a_star(start, goal, obstacles, (6,6))
# print(path)

# for i in range(len(path)-1):
#     coord1 = path[i]
#     coord2 = path[i+1]
#     print(move_agent(coord1,coord2))