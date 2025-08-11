"""
operator_funcs.py
=================

This module contains **pure, stateless helper functions** for spatial reasoning
and one-step action planning in the Melting Pot environments.

It connects to other parts of the repo as follows:
- Agents (like `SelfishAgent`) can use these helpers to decide their next move.
- The calibration step in `run_agents.py` uses `get_north` to orient agents.
- The environment runner uses `get_movement_actions` to translate
  "where I am" + "where I want to go" into **integer action IDs**.

Key ideas:
- All functions here avoid side effects (they don’t touch environment state directly).
- Movement is expressed **first** in string tokens (e.g. `"FORWARD"`) then
  converted to integer IDs via an externally provided `action_map` (from `base_agent.py`).
"""

import heapq
import numpy as np
from typing import Dict, Tuple, Set, List

# Type alias for coordinates on the grid
Coord = Tuple[int, int]

# ---------------------------------------------------------------------------
# Pathfinding: A*
# ---------------------------------------------------------------------------
def a_star(start: Coord, goal: Coord, obstacles: Set[Coord], grid_size: Tuple[int, int]) -> List[Coord]:
    """
    Compute an A* path from `start` to `goal`, avoiding any coordinates in `obstacles`.
    Returns the path as a list of coordinates, including start and goal.
    If no path exists, returns an empty list.

    This is used by both:
      - get_movement_actions (to plan 1-step moves)
      - Agents that want to chase objects while avoiding walls/trees.
    """
    if start == goal:
        return [start]

    def heuristic(a: Coord, b: Coord) -> int:
        """Manhattan distance: minimal steps without diagonals."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from: Dict[Coord, Coord] = {}
    g_score = {start: 0}

    rows, cols = grid_size

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path by walking backward from goal
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nb = (current[0] + dx, current[1] + dy)
            # Skip out-of-bounds or blocked tiles
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and nb not in obstacles:
                tentative = g_score[current] + 1
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))

    return []  # No path found


# ---------------------------------------------------------------------------
# Orientation helper
# ---------------------------------------------------------------------------
def get_north(orientation: str):
    """
    Rotate the agent one step toward 'north'.

    Returns:
      - action token string (e.g. "TURN_LEFT", "NOOP")
      - the new orientation string.

    This is incremental: if an agent is facing south,
    you'll need to call it multiple times on successive timesteps
    to fully rotate to north.
    """
    if orientation == "north":
        return "NOOP", "north"
    if orientation == "east":
        return "TURN_LEFT", "north"
    if orientation == "south":
        return "TURN_LEFT", "east"
    if orientation == "west":
        return "TURN_RIGHT", "north"
    return "NOOP", "north"


# ---------------------------------------------------------------------------
# Movement helper
# ---------------------------------------------------------------------------
def move_agent(coord_init: Coord, coord_final: Coord) -> str:
    """
    Given two adjacent coordinates (row, col), return the movement token
    required to go from init → final, assuming 'north' is up.

    Possible tokens: "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "NOOP"

    Special case:
    - If the move is diagonal or >1 step away, returns "NOOP".
    """
    r0, c0 = coord_init
    r1, c1 = coord_final

    # No movement
    if (r0, c0) == (r1, c1):
        return "NOOP"

    # Horizontal movement
    if r0 == r1 and c0 > c1:
        return "STEP_LEFT"
    if r0 == r1 and c0 < c1:
        return "STEP_RIGHT"

    # Vertical movement
    if r0 > r1 and c0 == c1:
        return "FORWARD"   # Up
    if r0 < r1 and c0 == c1:
        return "BACKWARD"  # Down

    # Not a valid 4-connected step
    return "NOOP"


# ---------------------------------------------------------------------------
# Multi-agent movement planner
# ---------------------------------------------------------------------------
def get_movement_actions(
    operator_output: Dict[str, Tuple[Coord, Coord, Set[Coord]]],
    colour_dict: List[str],
    num_players: int,
    grid_size: Tuple[int, int],
    action_map: Dict[str, int],
) -> np.ndarray:
    """
    Compute **one-step movement actions** for each player.

    Inputs:
      - operator_output[color] = (init_pos, goal_pos, obstacles_set)
        * Usually output of perception/converter logic.
      - colour_dict[i] = color string for player i (matches operator_output keys).
      - num_players = total number of agents.
      - grid_size = (rows, cols)
      - action_map: maps tokens (e.g. "FORWARD") to integer IDs.

    Output:
      - NumPy array of shape (num_players,), dtype=int
        where each entry is an integer action ID.

    Fallback: If a path can't be found, or input is missing, agent does "NOOP".
    """
    moves = np.zeros((num_players,), dtype=int)
    noop_id = action_map.get("NOOP", 0)

    for i in range(num_players):
        color = colour_dict[i]

        # If no data for this color → NOOP
        if color not in operator_output:
            moves[i] = noop_id
            continue

        init, goal, obstacles = operator_output[color]
        obstacles = set(obstacles) if obstacles is not None else set()

        # Plan path
        path = a_star(init, goal, obstacles, grid_size)

        if not path or len(path) < 2:
            moves[i] = noop_id
            continue

        # First step in path → token → integer ID
        step_token = move_agent(path[0], path[1])
        moves[i] = action_map.get(step_token, noop_id)

    return moves
