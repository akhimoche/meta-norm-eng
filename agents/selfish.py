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
#   - Uses utils/operator_funcs.a_star (or a_star_with_norms) for pathfinding.
# --------------------------------------------------------------------

from .base_agent import BaseAgent, ACTION_MAP
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from utils.operator_funcs import a_star as op_a_star, a_star_with_norms
from utils.norms.growth_reserve_local import GrowthReserveLocal

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
        seed: int | None = None,
        # ---- optional norm wiring (defaults keep behavior identical to before) ----
        use_norms: bool = False,          # enable the soft local-reserve norm
        reserve_K: int = 3,               # K_local: keep >= 3 apples within L2 radius
        reserve_radius: float = 2.0,      # L2 radius used by the substrate
        norm_penalty: float = 5.0,        # step penalty when breaching the rule
        epsilon: float = 0.0              # ε-compliance: prob. to ignore soft penalty
    ):
        # Call BaseAgent constructor to set RNG, action map, etc.
        super().__init__(agent_id, seed=seed, action_map=ACTION_MAP)
        # Store agent-specific info
        self.agent_id = agent_id
        self.action_min = int(action_min)
        self.action_max = int(action_max)
        self.converter = converter
        self.color = color

        # --- Optional norm (soft local reserve: leave >= K apples within L2=radius) ---
        self.use_norms = bool(use_norms)
        self.norm = GrowthReserveLocal(
            radius=reserve_radius, K=reserve_K, penalty=norm_penalty, seed=seed or 0
        )
        self.norm.set_epsilon(str(self.id), float(epsilon))

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

        # Step 3: Find apples (prefer exact live label; fallback to substring)
        apples = state.get("apple", [])
        if not apples:
            # fallback, but exclude any 'wait' variants
            apples = self._collect_positions(state, lambda k: ("apple" in k) and ("wait" not in k.lower()))

        # Step 4: Build obstacle set
        walls  = set(self._collect_positions(state, lambda k: "wall" in k))
        trees  = set(self._collect_positions(state, lambda k: "tree" in k))
        obstacles: Set[Tuple[int, int]] = walls | trees

        # Step 5: Compute grid size
        grid_size = self._grid_size(state)

        # (Norm hook) keep the norm's apple set up to date
        self.norm.update_apples(set(apples))

         # Step 6: Find best apple by effective cost = steps + expected penalty
        best_path = None
        best_score = float("inf")
        eid = str(self.id)

        for goal in apples:
            # If norms are ON and ε == 0, skip apples that would breach K.
            # This makes ε=0 behave "almost hard" so the effect is obvious.
            if self.use_norms and self.norm.epsilon_by_agent.get(eid, 0.0) == 0.0:
                if self.norm.would_breach(goal):
                    continue

            # Plan a path to this apple
            if self.use_norms:
                path = a_star_with_norms(
                    start, goal, obstacles, grid_size,
                    agent_id=eid,
                    norms_active=True,
                    norms_blocked=self.norm.is_blocked,     # always False in soft version
                    norms_penalty=self.norm.step_penalty
                )
            else:
                path = op_a_star(start, goal, obstacles, grid_size)

            # Treat [] or a 1-node path as "no usable path"
            if not path or len(path) < 2:
                continue

            # Base cost = number of steps (edges)
            base_steps = len(path) - 1

            # Add expected penalty for the *final step* into the apple (deterministic)
            if self.use_norms:
                exp_pen = self.norm.expected_step_penalty(eid, path[-2], path[-1])
                score = base_steps + exp_pen
            else:
                score = base_steps

            # Keep the best (lowest) effective score
            if score < best_score:
                best_score = score
                best_path = path

        # Step 7: No reachable apple → fallback random move
        if not best_path or len(best_path) < 2:
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