# --------------------------------------------------------------------
# SelfishAgent — A hard-coded heuristic agent for Melting Pot
# --------------------------------------------------------------------
# PURPOSE:
#   - Represents a single agent that tries to collect the *nearest reachable apple*
#     using A* pathfinding, avoiding obstacles.
#   - Does NOT learn; instead follows a fixed algorithm.
#   - Assumes agents have been "calibrated" to face NORTH before the main loop starts.
#
# CONNECTIONS TO OTHER FILES:
#   - Inherits from BaseAgent (agents/base_agent.py) to get:
#       * self.id          — agent identifier
#       * self.rng         — per-agent random number generator
#       * self.action_map  — mapping from action tokens to integer IDs
#   - Instantiated in scripts/run_agents.py:
#       * Given action range (a_min, a_max), the converter object (for pixel → symbol parsing),
#         and its unique color label (from calibration).
#   - Uses the converter from env/mp_llm_env.py to convert the environment’s RGB frame
#     into a symbolic "state" (dictionary of object labels → positions).
#   - Does NOT directly call utils/operator_funcs — pathfinding logic (_a_star) is implemented locally.
# --------------------------------------------------------------------

from .base_agent import BaseAgent, ACTION_MAP
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# Translation-only actions (move in the grid without turning or zapping).
# These correspond to:
#   1 = FORWARD, 2 = BACKWARD, 3 = STEP_LEFT, 4 = STEP_RIGHT
FALLBACK_TRANSLATIONS = np.array([1, 2, 3, 4], dtype=int)

class SelfishAgent(BaseAgent):
    """
    Heuristic agent that:
      - Finds the nearest reachable apple using A* search.
      - Takes the first step along the optimal path.
      - If no apple is reachable, chooses a random movement action.
    """

    def __init__(
        self,
        agent_id: int,
        action_min: int,
        action_max: int,
        converter,       # Object that can turn RGB frames into symbolic states
        color: str,      # Agent's color label from calibration (e.g., "red")
        seed: int | None = None
    ):
        # Call BaseAgent constructor to set RNG, action map, etc.
        super().__init__(agent_id, seed=seed, action_map=ACTION_MAP)
        # Store agent-specific info
        self.agent_id = agent_id
        self.action_min = int(action_min)
        self.action_max = int(action_max)
        self.converter = converter
        self.color = color

    # ----------------------------------------------------------------
    # HELPER: Convert observation frame → symbolic state dictionary
    # ----------------------------------------------------------------
    def _symbolic_state(self, obs) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extracts the 'WORLD.RGB' frame from the observation and
        converts it into a symbolic state using the converter.

        Returns:
            state: dict mapping object labels (e.g., "apple", "wall", "p_red_north")
                   to lists of (x, y) positions on the grid.
        """
        frame = obs.observation[0]["WORLD.RGB"]
        state = self.converter.image_to_state(frame)["global"]
        return state

    # ----------------------------------------------------------------
    # HELPER: Collect positions for labels matching a condition
    # ----------------------------------------------------------------
    @staticmethod
    def _collect_positions(state: Dict[str, List[Tuple[int, int]]], key_pred) -> List[Tuple[int, int]]:
        """
        Loops over all labels in the symbolic state and collects positions
        for those whose label satisfies `key_pred` (a boolean test function).
        """
        out: List[Tuple[int, int]] = []
        for k, v in state.items():
            if key_pred(k):
                out.extend(v)
        return out

    # ----------------------------------------------------------------
    # HELPER: Calculate grid size (width, height) from state
    # ----------------------------------------------------------------
    @staticmethod
    def _grid_size(state: Dict[str, List[Tuple[int, int]]]) -> Tuple[int, int]:
        """
        Determines the grid dimensions from all positions in the state.
        Needed for bounds-checking in A*.
        """
        max_x = 0
        max_y = 0
        for positions in state.values():
            for x, y in positions:
                if x > max_x: max_x = x
                if y > max_y: max_y = y
        return (max_x + 1, max_y + 1)  # +1 because coordinates are 0-indexed

    # ----------------------------------------------------------------
    # HELPER: A* pathfinding implementation
    # ----------------------------------------------------------------
    @staticmethod
    def _a_star(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid_size: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Finds a shortest path from start to goal using 4-directional A* search.
        Returns the path as a list of coordinates [start, ..., goal],
        or None if no path exists.
        """
        import heapq
        def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan distance

        w, hgt = grid_size
        nbrs = [(1,0), (-1,0), (0,1), (0,-1)]  # Right, Left, Down, Up
        openh = [(h(start, goal), start)]
        g = {start: 0}  # cost-so-far
        came = {}
        closed = set()

        while openh:
            _, cur = heapq.heappop(openh)
            if cur == goal:
                # Reconstruct path
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            if cur in closed:
                continue
            closed.add(cur)

            for dx, dy in nbrs:
                nx, ny = cur[0]+dx, cur[1]+dy
                if not (0 <= nx < w and 0 <= ny < hgt): continue
                n = (nx, ny)
                if n in obstacles: continue
                cand = g[cur] + 1
                if cand < g.get(n, 1e9):
                    g[n] = cand
                    came[n] = cur
                    f = cand + h(n, goal)
                    heapq.heappush(openh, (f, n))
        return None

    # ----------------------------------------------------------------
    # MAIN: Decide what action to take this step
    # ----------------------------------------------------------------
    def act(self, obs):
        """
        Main decision loop:
          1. Parse the observation into a symbolic state.
          2. Find own position by matching color label.
          3. Identify all apple positions.
          4. Identify obstacles (walls, trees).
          5. Run A* to find nearest reachable apple.
          6. If no path exists, move randomly.
          7. If path exists, take the first step toward the apple.
        """
        # Step 1: Get symbolic map of the world
        state = self._symbolic_state(obs)

        # Step 2: Locate this agent's position by its color label
        desired_prefix = f"p_{self.color}_"
        starts_any = self._collect_positions(state, lambda k: k.startswith(desired_prefix))
        start: Optional[Tuple[int, int]] = starts_any[0] if starts_any else None

        # If we can't find ourselves → move randomly
        if start is None:
            return int(self.rng.choice(FALLBACK_TRANSLATIONS))

        # Step 3: Find apples
        apples = self._collect_positions(state, lambda k: "apple" in k)

        # Step 4: Build obstacle set
        walls  = set(self._collect_positions(state, lambda k: "wall" in k))
        trees  = set(self._collect_positions(state, lambda k: "tree" in k))
        obstacles: Set[Tuple[int, int]] = walls | trees

        # Step 5: Compute grid size
        grid_size = self._grid_size(state)

        # Step 6: Find nearest reachable apple with A*
        best_path = None
        best_len = 10**9
        for goal in apples:
            path = self._a_star(start, goal, obstacles, grid_size)
            if path is not None and len(path) < best_len:
                best_len = len(path)
                best_path = path

        # Step 7: No reachable apple → fallback random move
        if best_path is None or len(best_path) < 2:
            return int(self.rng.choice(FALLBACK_TRANSLATIONS))

        # Step 8: Take the first step toward the apple
        next_step = best_path[1]
        dx = next_step[0] - start[0]
        dy = next_step[1] - start[1]

        # Map delta (dx, dy) to discrete action ID
        if dy == -1:   action_id = 1  # FORWARD (north)
        elif dy == 1:  action_id = 2  # BACKWARD (south)
        elif dx == -1: action_id = 3  # STEP_LEFT (west)
        elif dx == 1:  action_id = 4  # STEP_RIGHT (east)
        else:          action_id = int(self.rng.choice(FALLBACK_TRANSLATIONS))

        return action_id

