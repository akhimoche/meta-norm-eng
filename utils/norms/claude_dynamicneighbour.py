"""
Dynamic Density-Based Neighbor Protection Norm for Commons Harvest.

This norm dynamically adjusts protection based on local apple density,
protecting vulnerable (low-density) positions more frequently than
robust (high-density) positions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from utils.norms.norm import Norm, Coord


class DynamicDensityNorm(Norm):
    """
    Dynamic Density-Based Protection norm for Commons Harvest (Open layout).

    Summary
    -------
    Adapts protection frequency based on local apple density. Positions with
    fewer neighbors (more vulnerable) are protected more frequently, while
    positions with many neighbors (more robust) are protected less often.
    This creates a density-gradient defense that prioritizes vulnerable areas.

    Why this works at high epsilon (0.5-0.9)
    ----------------------------------------
    In Commons Harvest, patches collapse when local density drops below the
    regrowth threshold. This typically happens at patch EDGES and TIPS first,
    because these positions have fewer natural neighbors.

    This norm addresses this by:
    1. **Vulnerability Assessment**: Positions with few neighbors are "vulnerable"
       and receive protection more often
    2. **Redundancy Exploitation**: Positions with many neighbors are "robust" and
       can tolerate less protection because their neighbors provide backup
    3. **Dynamic Adaptation**: Protection shifts over time, ensuring all positions
       get some coverage while prioritizing the most vulnerable

    At high epsilon, agents will still eat some protected positions. But:
    - Vulnerable edges/tips are protected more often, so they survive longer
    - When violations do happen, the density-weighted protection means core
      positions remain to seed regrowth
    - The dynamic cycling ensures no position is permanently neglected

    How it works
    ------------
    1. Computes "vulnerability score" for each position: fewer neighbors = higher score
    2. Positions are assigned to protection tiers based on vulnerability
    3. High-vulnerability positions appear in MORE protection patterns
    4. Low-vulnerability positions appear in FEWER protection patterns
    5. Pattern selection cycles over time

    Hyperparameters
    ---------------
    PERIOD : int
        Timesteps per pattern cycle. Default: 20.
    NUM_VULNERABILITY_TIERS : int
        Number of tiers to divide positions into. Default: 4.
    BASE_PROTECTION_PROB : float
        Minimum protection probability for most robust tier. Default: 0.3.
    MAX_PROTECTION_PROB : float
        Maximum protection probability for most vulnerable tier. Default: 0.9.
    REGROWTH_RADIUS : float
        L2 distance for neighbor counting. Default: 2.0.
    """

    # ---- Hyperparameters ----
    PERIOD: int = 20
    NUM_VULNERABILITY_TIERS: int = 4
    BASE_PROTECTION_PROB: float = 0.3  # Robust positions: protected 30% of time
    MAX_PROTECTION_PROB: float = 0.9   # Vulnerable positions: protected 90% of time
    REGROWTH_RADIUS: float = 2.0

    # All apple positions in the environment
    APPLE_POSITIONS: Set[Coord] = {
        # Upper left corner patch (6 apples)
        (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3),
        
        # Lower left diamond patch (13 apples)
        (3, 6), (2, 7), (3, 7), (4, 7), 
        (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
        (2, 9), (3, 9), (4, 9), (3, 10),
        
        # Upper mid-left diamond patch (13 apples)
        (8, 1), (7, 2), (8, 2), (9, 2),
        (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
        (7, 4), (8, 4), (9, 4), (8, 5),
        
        # Upper mid-right diamond patch (13 apples)
        (15, 1), (14, 2), (15, 2), (16, 2),
        (13, 3), (14, 3), (15, 3), (16, 3), (17, 3),
        (14, 4), (15, 4), (16, 4), (15, 5),
        
        # Lower right diamond patch (13 apples)
        (20, 6), (19, 7), (20, 7), (21, 7),
        (18, 8), (19, 8), (20, 8), (21, 8), (22, 8),
        (19, 9), (20, 9), (21, 9), (20, 10),
        
        # Upper right corner patch (6 apples)
        (20, 1), (21, 1), (22, 1), (21, 2), (22, 2), (22, 3),
    }

    # Spawn points that should never be blocked
    SPAWN_POINTS: Set[Coord] = {(7, 7), (16, 7)}

    def __init__(self, epsilon: float = 0.0):
        super().__init__("dynamic_density", epsilon)
        
        # Precompute neighbor relationships
        self._neighbors: Dict[Coord, Set[Coord]] = self._compute_neighbors()
        
        # Compute vulnerability scores (inverse of neighbor count)
        self._vulnerability: Dict[Coord, float] = self._compute_vulnerability()
        
        # Assign positions to vulnerability tiers
        self._tiers: List[Set[Coord]] = self._assign_tiers()
        
        # Precompute protection patterns for each phase
        self._num_phases = 10  # 10 different phases to cycle through
        self._patterns: List[Set[Coord]] = self._generate_density_patterns()
        
        # Add P spawn points
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
        """Compute L2 (Euclidean) distance between two positions."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _compute_neighbors(self) -> Dict[Coord, Set[Coord]]:
        """Precompute which positions are neighbors (within L2 regrowth radius)."""
        neighbors: Dict[Coord, Set[Coord]] = {}
        
        for pos in self.APPLE_POSITIONS:
            neighbors[pos] = set()
            for other in self.APPLE_POSITIONS:
                if pos != other and self._l2_distance(pos, other) <= self.REGROWTH_RADIUS:
                    neighbors[pos].add(other)
        
        return neighbors

    def _compute_vulnerability(self) -> Dict[Coord, float]:
        """
        Compute vulnerability score for each position.
        
        Vulnerability = 1 / (1 + neighbor_count)
        
        Positions with fewer neighbors are more vulnerable and get higher scores.
        """
        vulnerability: Dict[Coord, float] = {}
        
        for pos in self.APPLE_POSITIONS:
            neighbor_count = len(self._neighbors[pos])
            # Inverse relationship: fewer neighbors = higher vulnerability
            vulnerability[pos] = 1.0 / (1.0 + neighbor_count)
        
        return vulnerability

    def _assign_tiers(self) -> List[Set[Coord]]:
        """
        Assign positions to vulnerability tiers.
        
        Tier 0: Most vulnerable (fewest neighbors)
        Tier N-1: Most robust (most neighbors)
        """
        # Sort positions by vulnerability (highest first)
        sorted_positions = sorted(
            self.APPLE_POSITIONS,
            key=lambda p: self._vulnerability[p],
            reverse=True
        )
        
        # Divide into tiers
        tiers: List[Set[Coord]] = [set() for _ in range(self.NUM_VULNERABILITY_TIERS)]
        positions_per_tier = len(sorted_positions) // self.NUM_VULNERABILITY_TIERS
        
        for i, pos in enumerate(sorted_positions):
            tier_idx = min(i // max(1, positions_per_tier), self.NUM_VULNERABILITY_TIERS - 1)
            tiers[tier_idx].add(pos)
        
        return tiers

    def _get_tier_protection_prob(self, tier_idx: int) -> float:
        """
        Get protection probability for a tier.
        
        Tier 0 (most vulnerable) gets MAX_PROTECTION_PROB
        Tier N-1 (most robust) gets BASE_PROTECTION_PROB
        """
        if self.NUM_VULNERABILITY_TIERS <= 1:
            return (self.BASE_PROTECTION_PROB + self.MAX_PROTECTION_PROB) / 2
        
        # Linear interpolation from MAX (tier 0) to BASE (tier N-1)
        fraction = tier_idx / (self.NUM_VULNERABILITY_TIERS - 1)
        return self.MAX_PROTECTION_PROB - fraction * (self.MAX_PROTECTION_PROB - self.BASE_PROTECTION_PROB)

    def _generate_density_patterns(self) -> List[Set[Coord]]:
        """
        Generate protection patterns that respect density-based probabilities.
        
        Each pattern includes positions based on their tier's protection probability,
        with variation introduced by phase-specific pseudo-random selection.
        """
        patterns: List[Set[Coord]] = []
        
        for phase in range(self._num_phases):
            pattern: Set[Coord] = set()
            
            for tier_idx, tier_positions in enumerate(self._tiers):
                protection_prob = self._get_tier_protection_prob(tier_idx)
                
                for pos in tier_positions:
                    # Deterministic selection based on position, phase, and tier
                    hash_val = (pos[0] * 73 + pos[1] * 179 + phase * 31 + tier_idx * 17) % 100
                    
                    # Compare hash to protection probability threshold
                    if hash_val < protection_prob * 100:
                        pattern.add(pos)
            
            # Ensure pattern isn't empty or nearly empty
            if len(pattern) < 20:
                # Add more positions from most vulnerable tier
                for pos in self._tiers[0]:
                    pattern.add(pos)
            
            patterns.append(pattern)
        
        return patterns

    def _ensure_neighbor_threshold(self, pattern: Set[Coord]) -> Set[Coord]:
        """
        Adjust pattern to ensure every harvestable position has at least 3 protected neighbors.
        
        This is a soft guarantee - we add positions if needed but don't remove any.
        """
        result = set(pattern)
        harvestable = self.APPLE_POSITIONS - result
        
        for pos in harvestable:
            protected_neighbors = len(self._neighbors[pos] & result)
            
            if protected_neighbors < 3:
                # Add some of this position's neighbors to protection
                neighbors_to_add = list(self._neighbors[pos] - result)
                needed = 3 - protected_neighbors
                
                # Prioritize adding high-vulnerability neighbors
                neighbors_to_add.sort(
                    key=lambda n: self._vulnerability.get(n, 0),
                    reverse=True
                )
                
                for neighbor in neighbors_to_add[:needed]:
                    result.add(neighbor)
        
        return result

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the set of blocked (protected) coordinates at timestep t.
        
        Uses density-weighted patterns where vulnerable positions (few neighbors)
        are protected more frequently than robust positions (many neighbors).
        
        Args:
            t: Current simulation timestep (0-based)
            
        Returns:
            Set of coordinates that should be treated as obstacles
        """
        # Determine which pattern to use
        phase = (t // self.PERIOD) % self._num_phases
        base_pattern = self._patterns[phase]
        
        # Ensure minimum neighbor threshold for regrowth
        blocked = self._ensure_neighbor_threshold(base_pattern)
        
        # Never block spawn points
        blocked -= self._spawns
        
        return blocked


# Meta-norm documentation
meta_norm = {
    "verbal_explanation": (
        "The Dynamic Density norm adapts protection based on local apple density. "
        "Positions with fewer neighbors (patch edges, tips) are considered 'vulnerable' "
        "and are protected more frequently (up to 90% of the time). Positions with many "
        "neighbors (patch cores) are considered 'robust' and are protected less often "
        "(as low as 30% of the time). This creates a density-gradient defense that "
        "prioritizes protecting the most at-risk positions while allowing harvesting "
        "from naturally robust areas. The pattern cycles every 20 timesteps across "
        "10 different phases, ensuring all positions get coverage over time."
    ),
    "reasoning": (
        "Commons Harvest patches typically collapse from the edges inward. This happens "
        "because edge positions have fewer natural neighbors, so when they're eaten, "
        "regrowth probability drops faster. This norm addresses the root cause:\n\n"
        "1. **Vulnerability Assessment**: Each position is scored based on neighbor count. "
        "Tips and edges (1-2 neighbors) score high, cores (5-6 neighbors) score low.\n\n"
        "2. **Proportional Protection**: Vulnerable positions are protected 60-90% of time. "
        "Robust positions are protected only 30-50%. This is optimal because:\n"
        "   - Vulnerable positions NEED protection (low natural redundancy)\n"
        "   - Robust positions CAN tolerate harvesting (high natural redundancy)\n\n"
        "3. **Dynamic Cycling**: 10 different patterns cycle through, each respecting the "
        "density-weighted probabilities but with variation. This means:\n"
        "   - No position is always protected or always exposed\n"
        "   - High-epsilon violations are spread across different positions over time\n"
        "   - Recovery opportunities exist for all areas\n\n"
        "4. **Neighbor Threshold Guarantee**: Each pattern is adjusted to ensure every "
        "harvestable position has at least 3 protected neighbors, maintaining the 2.5% "
        "regrowth rate even when agents harvest from robust areas.\n\n"
        "At high epsilon (0.5-0.9), the density weighting ensures that violations "
        "disproportionately affect robust positions that can recover, while vulnerable "
        "positions remain protected more often."
    ),
    "code_with_placeholders": "See DynamicDensityNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "PERIOD": 20,
        "NUM_VULNERABILITY_TIERS": 4,
        "BASE_PROTECTION_PROB": 0.3,
        "MAX_PROTECTION_PROB": 0.9,
        "REGROWTH_RADIUS": 2.0,
        "notes": (
            "The 4-tier system divides 64 positions into ~16 per tier. Tier 0 (most vulnerable, "
            "edges/tips) is protected 90% of the time. Tier 3 (most robust, cores) is protected "
            "30% of the time. This creates a ~60% protection gradient from edge to core. "
            "The 10 phases ensure good coverage diversity while maintaining tier probabilities."
        )
    }
}
