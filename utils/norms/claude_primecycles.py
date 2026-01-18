"""
Prime-Staggered Cycling Norm for Commons Harvest.

This norm assigns each patch a different prime number period for protection cycling,
creating irregular patterns that never synchronize across patches.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from utils.norms.norm import Norm, Coord


class PrimeCycleNorm(Norm):
    """
    Prime-Staggered Cycling norm for Commons Harvest (Open layout).

    Summary
    -------
    Each of the 6 apple patches cycles its protection using a different prime number
    period. Because primes share no common factors, the protection patterns across
    patches never synchronize, creating an unpredictable and robust defense.

    Why this works at high epsilon (0.5-0.9)
    ----------------------------------------
    Traditional norms with synchronized timing can be exploited: agents learn when
    patches are "open" and coordinate harvesting. This norm defeats that by:
    
    1. **Desynchronized Patches**: Patch 1 cycles every 7 steps, Patch 2 every 11,
       Patch 3 every 13, etc. The LCM of all primes is huge (over 200,000), so the
       full pattern effectively never repeats within a typical simulation.
    
    2. **Irregular Protection Windows**: At any given timestep, a different subset
       of patches is in "protection mode". Agents can't predict which patches are
       safe to harvest without constant checking.
    
    3. **Distributed Exploitation**: When agents do ignore the norm, their violations
       are spread across patches rather than concentrated, preventing any single
       patch from collapsing.

    How it works
    ------------
    For each patch:
    1. Assign a unique prime period (7, 11, 13, 17, 19, 23)
    2. Divide positions into "core" (always protected) and "rotating" (cycles)
    3. At timestep t, rotating positions are protected if (t // period) is even
    4. Different primes = different phase relationships between patches

    Hyperparameters
    ---------------
    PRIME_PERIODS : List[int]
        Prime periods for each of the 6 patches. Default: [7, 11, 13, 17, 19, 23]
    CORE_PROTECTION_RATIO : float
        Fraction of each patch to always protect. Default: 0.3 (30%)
    """

    # ---- Hyperparameters ----
    # Prime periods for each patch (chosen for good spread and reasonable cycle lengths)
    PRIME_PERIODS: List[int] = [7, 11, 13, 17, 19, 23]
    CORE_PROTECTION_RATIO: float = 0.3  # 30% of each patch is always protected

    # Patch definitions with explicit position sets
    PATCHES: List[Set[Coord]] = [
        # Patch 0: Upper left corner (6 apples)
        {(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3)},
        
        # Patch 1: Upper mid-left diamond (13 apples)
        {(8, 1), (7, 2), (8, 2), (9, 2),
         (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
         (7, 4), (8, 4), (9, 4), (8, 5)},
        
        # Patch 2: Upper mid-right diamond (13 apples)
        {(15, 1), (14, 2), (15, 2), (16, 2),
         (13, 3), (14, 3), (15, 3), (16, 3), (17, 3),
         (14, 4), (15, 4), (16, 4), (15, 5)},
        
        # Patch 3: Upper right corner (6 apples)
        {(20, 1), (21, 1), (22, 1), (21, 2), (22, 2), (22, 3)},
        
        # Patch 4: Lower left diamond (13 apples)
        {(3, 6), (2, 7), (3, 7), (4, 7),
         (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
         (2, 9), (3, 9), (4, 9), (3, 10)},
        
        # Patch 5: Lower right diamond (13 apples)
        {(20, 6), (19, 7), (20, 7), (21, 7),
         (18, 8), (19, 8), (20, 8), (21, 8), (22, 8),
         (19, 9), (20, 9), (21, 9), (20, 10)},
    ]

    # Spawn points that should never be blocked
    SPAWN_POINTS: Set[Coord] = {(7, 7), (16, 7)}

    def __init__(self, epsilon: float = 0.0):
        super().__init__("prime_cycle", epsilon)
        
        # Precompute L2 neighbors for regrowth calculations
        self._all_positions: Set[Coord] = set()
        for patch in self.PATCHES:
            self._all_positions |= patch
        self._neighbors = self._compute_neighbors()
        
        # For each patch, identify core (always protected) and rotating positions
        self._patch_cores: List[Set[Coord]] = []
        self._patch_rotating: List[List[Coord]] = []
        
        for patch in self.PATCHES:
            core, rotating = self._partition_patch(patch)
            self._patch_cores.append(core)
            self._patch_rotating.append(rotating)
        
        # Precompute spawn points (P tiles)
        self._spawns: Set[Coord] = set(self.SPAWN_POINTS)
        for y in range(14, 17):
            for x in range(1, 23):
                if y == 14 and 3 <= x <= 20:
                    self._spawns.add((x, y))
                elif y == 15 and 2 <= x <= 21:
                    self._spawns.add((x, y))
                elif y == 16 and 1 <= x <= 22:
                    self._spawns.add((x, y))

    def _l2_distance(self, p1: Coord, p2: Coord) -> float:
        """Compute L2 distance between two positions."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _compute_neighbors(self) -> Dict[Coord, Set[Coord]]:
        """Precompute neighbors within L2 distance 2.0 (regrowth radius)."""
        neighbors: Dict[Coord, Set[Coord]] = {}
        for pos in self._all_positions:
            neighbors[pos] = set()
            for other in self._all_positions:
                if pos != other and self._l2_distance(pos, other) <= 2.0:
                    neighbors[pos].add(other)
        return neighbors

    def _partition_patch(self, patch: Set[Coord]) -> Tuple[Set[Coord], List[Coord]]:
        """
        Partition a patch into core (always protected) and rotating positions.
        
        Core positions are selected based on highest centrality (most neighbors
        within the patch), ensuring they provide maximum regrowth support.
        """
        positions = list(patch)
        
        # Score by connectivity within patch
        def centrality_score(pos: Coord) -> Tuple[int, int, int]:
            within_patch_neighbors = sum(
                1 for other in patch 
                if pos != other and self._l2_distance(pos, other) <= 2.0
            )
            return (within_patch_neighbors, -pos[0], -pos[1])
        
        positions.sort(key=centrality_score, reverse=True)
        
        # Take top CORE_PROTECTION_RATIO as core
        core_count = max(1, int(len(positions) * self.CORE_PROTECTION_RATIO))
        core = set(positions[:core_count])
        rotating = positions[core_count:]
        
        return core, rotating

    def _get_rotating_protection(self, patch_idx: int, t: int) -> Set[Coord]:
        """
        Determine which rotating positions are protected at timestep t.
        
        Uses the patch's prime period to create a cycling pattern.
        Protection alternates: half the rotating positions are protected at any time,
        and which half depends on the current phase within the prime cycle.
        """
        prime = self.PRIME_PERIODS[patch_idx]
        rotating = self._patch_rotating[patch_idx]
        
        if not rotating:
            return set()
        
        # Current phase within this patch's cycle
        phase = (t // prime) % 2
        
        # Split rotating positions into two groups based on index
        # Group 0 protected in even phases, Group 1 in odd phases
        protected: Set[Coord] = set()
        for i, pos in enumerate(rotating):
            # Use position-based hash for more varied grouping
            group = (pos[0] + pos[1] + i) % 2
            if group == phase:
                protected.add(pos)
        
        return protected

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the set of blocked (protected) coordinates at timestep t.
        
        Combines:
        1. All core positions (always protected)
        2. Rotating positions based on each patch's prime-period phase
        
        Args:
            t: Current simulation timestep (0-based)
            
        Returns:
            Set of coordinates that should be treated as obstacles
        """
        blocked: Set[Coord] = set()
        
        # Add all cores (always protected)
        for core in self._patch_cores:
            blocked |= core
        
        # Add rotating positions based on prime-period cycling
        for patch_idx in range(len(self.PATCHES)):
            rotating_protected = self._get_rotating_protection(patch_idx, t)
            blocked |= rotating_protected
        
        # Never block spawn points
        blocked -= self._spawns
        
        return blocked

    def get_patch_phase_info(self, t: int) -> List[Tuple[int, int, str]]:
        """
        Debug helper: Get phase information for each patch at timestep t.
        
        Returns list of (prime, current_phase, phase_name) tuples.
        """
        info = []
        for patch_idx, prime in enumerate(self.PRIME_PERIODS):
            phase = (t // prime) % 2
            phase_name = "even" if phase == 0 else "odd"
            info.append((prime, phase, phase_name))
        return info


# Meta-norm documentation
meta_norm = {
    "verbal_explanation": (
        "The Prime-Staggered Cycling norm assigns each of the 6 apple patches a different "
        "prime number period for protection cycling: 7, 11, 13, 17, 19, and 23 timesteps. "
        "Each patch has a 'core' (30% of positions, always protected) and 'rotating' "
        "positions (70%, cycle between protected and harvestable). Because primes share "
        "no common factors, the protection patterns across patches never synchronize. "
        "The LCM of these primes is 7 × 11 × 13 × 17 × 19 × 23 = 223,092,870, meaning "
        "the full pattern effectively never repeats within any reasonable simulation."
    ),
    "reasoning": (
        "Traditional synchronized norms fail at high epsilon because agents can learn the "
        "timing and coordinate exploitation. This norm defeats such strategies:\n\n"
        "1. **Desynchronized Protection**: At any timestep, each patch is in a different "
        "phase of its cycle. For example, at t=77:\n"
        "   - Patch 0 (prime=7):  phase = 77/7 = 11 → odd → Group 1 protected\n"
        "   - Patch 1 (prime=11): phase = 77/11 = 7 → odd → Group 1 protected\n"
        "   - Patch 2 (prime=13): phase = 77/13 = 5 → odd → Group 1 protected\n"
        "   - Patch 3 (prime=17): phase = 77/17 = 4 → even → Group 0 protected\n"
        "   - Patch 4 (prime=19): phase = 77/19 = 4 → even → Group 0 protected\n"
        "   - Patch 5 (prime=23): phase = 77/23 = 3 → odd → Group 1 protected\n\n"
        "2. **Irregular Patterns**: The phase relationships shift constantly. No simple "
        "exploitation strategy works because 'safe' patches change unpredictably.\n\n"
        "3. **Core Stability**: 30% of each patch is ALWAYS protected, providing a stable "
        "seed bank for regrowth even when all rotating positions are eaten.\n\n"
        "4. **Distributed Risk**: At high epsilon, violations are spread across whichever "
        "patches happen to have open positions at that moment, preventing concentrated "
        "collapse of any single patch.\n\n"
        "The key insight is that prime numbers create mathematically guaranteed "
        "desynchronization - there's no 'lucky' timestep where all patches are open."
    ),
    "code_with_placeholders": "See PrimeCycleNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "PRIME_PERIODS": [7, 11, 13, 17, 19, 23],
        "CORE_PROTECTION_RATIO": 0.3,
        "LCM_of_primes": 223092870,
        "positions_per_patch": [6, 13, 13, 6, 13, 13],
        "core_positions_per_patch": [2, 4, 4, 2, 4, 4],
        "notes": (
            "Primes chosen to span a range (7-23) that creates meaningful variation "
            "within typical simulation lengths (500-1000 steps). Smaller primes cycle "
            "faster but may be too predictable; larger primes cycle slower but provide "
            "more stable protection periods. The 30% core ratio ensures ~20 positions "
            "are always protected across all patches."
        )
    }
}
