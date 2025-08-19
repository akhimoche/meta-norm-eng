# utils/norms/growth_reserve_local.py
from __future__ import annotations
from typing import Dict, Set, Tuple
import random
import math

Coord = Tuple[int, int]

class GrowthReserveLocal:
    """
    Soft local-reserve norm:
      - Uses L2 (Euclidean) radius (default 2.0) around a candidate apple tile.
      - If harvesting that apple would leave fewer than K uneaten apples within that radius,
        we add a step penalty (soft enforcement). Agents can still cross.
      - ε-compliance: per-agent probability to ignore the penalty on that step.

    Contract with planner:
      - Call update_apples(apples) each timestep before planning (apples are Coord set).
      - Use is_blocked(...) for hard enforcement (here: always False).
      - Use step_penalty(...) for soft enforcement.
    """

    def __init__(self, radius: float = 2.0, K: int = 3, penalty: float = 5.0, seed: int = 0):
        self.radius = float(radius)
        self.radius_sq = self.radius * self.radius
        self.K = int(K)
        self.penalty = float(penalty)
        self.apples: Set[Coord] = set()

        # ε per agent and RNGs per agent (seeded for reproducibility)
        self.epsilon_by_agent: Dict[str, float] = {}
        self._rng_by_agent: Dict[str, random.Random] = {}
        self._base_seed = int(seed) & 0xFFFFFFFF

    # --- External API ---

    def update_apples(self, apples: Set[Coord]) -> None:
        """Provide the set of currently-uneaten apple coordinates for this step."""
        self.apples = set(apples)

    def set_epsilon(self, agent_id: str, eps: float) -> None:
        """Set probability that the agent ignores soft penalties on a step."""
        self.epsilon_by_agent[agent_id] = max(0.0, min(1.0, float(eps)))

    # Hard enforcement (not used in the soft version)
    def is_blocked(self, agent_id: str, cell: Coord) -> bool:
        return False

    # Soft enforcement
    def step_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """
        Return extra step cost for moving cur->nxt.
        Applies only when nxt is currently an apple AND post-harvest neighbors < K.
        With prob ε(agent) the penalty is ignored for this step.
        """
        if nxt not in self.apples:
            return 0.0

        # Count neighbors that would remain AFTER harvesting nxt (exclude nxt itself)
        post_neighbors = self._count_neighbors_excluding(nxt)
        if post_neighbors >= self.K:
            return 0.0

        # ε-compliance: ignore soft penalty with prob ε(agent)
        eps = self.epsilon_by_agent.get(agent_id, 0.0)
        rng = self._rng_by_agent.get(agent_id)
        if rng is None:
            # Derive a per-agent seed in a simple, stable way
            derived = (self._base_seed ^ (hash(agent_id) & 0xFFFFFFFF)) & 0xFFFFFFFF
            rng = random.Random(derived)
            self._rng_by_agent[agent_id] = rng

        if rng.random() < eps:
            return 0.0  # ignore this step's penalty
        return self.penalty

    # --- Helpers ---

    def _count_neighbors_excluding(self, cell: Coord) -> int:
        """Number of apples within L2 radius (excluding the cell itself)."""
        r0, c0 = cell
        count = 0
        for (r, c) in self.apples:
            if (r, c) == (r0, c0):
                continue
            # L2 distance check: (dr^2 + dc^2) <= radius^2
            dr = r - r0
            dc = c - c0
            if (dr * dr + dc * dc) <= self.radius_sq + 1e-9:
                count += 1
        return count
# Add these inside the GrowthReserveLocal class (near the other methods)

    # --- Planning helpers (deterministic; no RNG consumed) ---
    def would_breach(self, cell: Coord) -> bool:
        """True if harvesting 'cell' would leave fewer than K neighbors within L2 radius."""
        return self._count_neighbors_excluding(cell) < self.K

    def expected_step_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """
        Expected (deterministic) penalty for planning:
          - 0 if nxt is not an apple or would not breach K.
          - (1 - ε(agent)) * penalty otherwise.
        """
        if nxt not in self.apples:
            return 0.0
        if not self.would_breach(nxt):
            return 0.0
        eps = self.epsilon_by_agent.get(agent_id, 0.0)
        return (1.0 - eps) * self.penalty
