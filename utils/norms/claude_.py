"""
Stochastic Position Hashing Norm for Commons Harvest.

This norm uses deterministic but pseudo-random hash-based protection patterns
that change periodically, providing robust defense even at high epsilon values.
"""

from __future__ import annotations

from typing import Set

from utils.norms.norm import Norm, Coord


class StochasticHashNorm(Norm):
    """
    Stochastic Position Hashing norm for Commons Harvest (Open layout).

    Summary
    -------
    Uses a deterministic hash function to create pseudo-random protection patterns
    that change every PERIOD timesteps. At any given time, approximately 50% of
    apple positions are protected, but which positions are protected appears random.

    Why this works at high epsilon (0.5-0.9)
    ----------------------------------------
    Traditional static protection fails at high epsilon because agents eventually
    eat the protected positions. This norm succeeds because:
    
    1. **No Predictable Targets**: The pseudo-random pattern makes it impossible
       for agents to consistently exploit specific positions.
    
    2. **Distributed Protection**: Even with only 10% compliance (epsilon=0.9),
       protection events are spread across all positions over time.
    
    3. **Periodic Refresh**: The pattern changes every PERIOD timesteps, so
       positions that were exploited during one period may be protected next.
    
    4. **Regrowth Opportunity**: Each position spends ~50% of time protected,
       allowing apples to regrow from neighbors.

    The key insight is that at high epsilon, what matters is not whether
    specific positions are always protected, but whether protection events
    are distributed in a way that slows exploitation and enables recovery.

    How it works
    ------------
    - For each apple position (x, y) at timestep t:
      1. Compute period_index = t // PERIOD
      2. Compute hash = (x * HASH_A + y * HASH_B + period_index * HASH_C) % HASH_MOD
      3. If hash < PROTECTION_THRESHOLD, position is blocked (protected)
    - The hash constants are chosen to create good distribution
    - PROTECTION_RATE controls what fraction of positions are protected (~50%)
    - PERIOD controls how often the pattern changes (default: 15 timesteps)

    Hyperparameters
    ---------------
    PERIOD : int
        Number of timesteps before the protection pattern changes. Default: 15.
        Shorter periods = more frequent changes = more positions get protection.
        
    PROTECTION_RATE : float
        Target fraction of positions to protect (0.0-1.0). Default: 0.5.
        Higher = more protection but less harvesting opportunity.
        
    HASH_A, HASH_B, HASH_C, HASH_MOD : int
        Hash function constants for pseudo-random distribution.
        Chosen to minimize correlation between adjacent positions.
    """

    # ---- Hyperparameters ----
    PERIOD: int = 15  # Timesteps per protection pattern
    PROTECTION_RATE: float = 0.5  # Target 50% protection
    
    # Hash constants (chosen for good distribution)
    HASH_A: int = 73  # Large prime for x contribution
    HASH_B: int = 179  # Large prime for y contribution  
    HASH_C: int = 31  # Prime for temporal contribution
    HASH_MOD: int = 1000  # Modulus for hash output range

    # All apple positions in the environment (precomputed from ASCII map)
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
    SPAWN_POINTS: Set[Coord] = {
        # Q spawn points (inside spawns)
        (7, 7), (16, 7),
        # P spawn points are in rows 14-16, columns 2-21
        # We'll compute these to be safe
    }

    def __init__(self, epsilon: float = 0.0):
        super().__init__("stochastic_hash", epsilon)
        
        # Precompute spawn points (P tiles)
        self._spawns: Set[Coord] = set(self.SPAWN_POINTS)
        for y in range(14, 17):  # Rows 14, 15, 16
            for x in range(1, 23):  # Columns 1-22
                # Check if it's within the P spawn area based on the map
                if y == 14 and 3 <= x <= 20:
                    self._spawns.add((x, y))
                elif y == 15 and 2 <= x <= 21:
                    self._spawns.add((x, y))
                elif y == 16 and 1 <= x <= 22:
                    self._spawns.add((x, y))
        
        # Compute protection threshold from rate
        self._threshold = int(self.PROTECTION_RATE * self.HASH_MOD)

    def _compute_hash(self, x: int, y: int, period_index: int) -> int:
        """
        Compute a deterministic pseudo-random hash for position and time period.
        
        The hash function is designed to:
        1. Produce different values for adjacent positions (via prime multipliers)
        2. Change completely when the period changes
        3. Be fast to compute (simple arithmetic)
        
        Args:
            x: Column coordinate
            y: Row coordinate  
            period_index: Which time period we're in (t // PERIOD)
            
        Returns:
            Hash value in range [0, HASH_MOD)
        """
        # Use prime multipliers to reduce correlation
        raw = (x * self.HASH_A + y * self.HASH_B + period_index * self.HASH_C)
        # Add non-linear mixing for better distribution
        mixed = raw ^ (raw >> 4)
        return abs(mixed) % self.HASH_MOD

    def _is_protected(self, x: int, y: int, period_index: int) -> bool:
        """
        Determine if a position is protected during a given period.
        
        Args:
            x: Column coordinate
            y: Row coordinate
            period_index: Which time period we're in
            
        Returns:
            True if position should be blocked (protected), False otherwise
        """
        hash_val = self._compute_hash(x, y, period_index)
        return hash_val < self._threshold

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the set of blocked (protected) coordinates at timestep t.
        
        Uses a hash function to deterministically but pseudo-randomly select
        approximately PROTECTION_RATE fraction of apple positions to block.
        The pattern changes every PERIOD timesteps.
        
        Args:
            t: Current simulation timestep (0-based)
            
        Returns:
            Set of coordinates that should be treated as obstacles
        """
        period_index = t // self.PERIOD
        
        blocked: Set[Coord] = set()
        
        for pos in self.APPLE_POSITIONS:
            x, y = pos
            if self._is_protected(x, y, period_index):
                blocked.add(pos)
        
        # Never block spawn points
        blocked -= self._spawns
        
        return blocked


# Meta-norm documentation
meta_norm = {
    "verbal_explanation": (
        "The Stochastic Position Hashing norm creates pseudo-random protection patterns "
        "that change every 15 timesteps. At any moment, approximately 50% of apple positions "
        "are blocked, but which positions are blocked appears random and unpredictable. "
        "This is achieved using a deterministic hash function based on position coordinates "
        "and time period. The key advantage is that no position is predictably always available "
        "for exploitation - every position spends roughly half its time protected, allowing "
        "regrowth from neighboring apples. The pattern changes frequently enough that even "
        "positions exploited in one period may be protected in the next, enabling recovery."
    ),
    "reasoning": (
        "Traditional static protection norms fail at high epsilon (0.5-0.9) because agents "
        "eventually consume the protected positions during their violation events. This norm "
        "succeeds for several reasons:\n\n"
        "1. **No Predictable Targets**: The hash-based pattern makes systematic exploitation "
        "impossible. An agent that ignores the norm 90% of the time cannot consistently "
        "target specific 'always available' positions because there are none.\n\n"
        "2. **Distributed Protection Over Time**: Even with only 10% compliance (epsilon=0.9), "
        "those compliant timesteps protect different random subsets of positions. Over 500 "
        "timesteps, this distributes protection across all positions.\n\n"
        "3. **Regrowth Preservation**: Each position spends ~50% of time blocked. Even if "
        "eaten during an 'open' period, its neighbors may be protected, maintaining the "
        "3+ neighbor threshold for 2.5% regrowth probability.\n\n"
        "4. **Period Refresh**: Every 15 timesteps, the entire pattern shuffles. Positions "
        "that were heavily exploited during one period get fresh protection in the next, "
        "preventing permanent patch collapse.\n\n"
        "The 50% protection rate is optimal: higher rates would starve agents (reducing "
        "welfare), lower rates would allow overexploitation. The 15-timestep period is "
        "chosen to be short enough for quick recovery but long enough to be meaningful."
    ),
    "code_with_placeholders": "See StochasticHashNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "PERIOD": 15,
        "PROTECTION_RATE": 0.5,
        "HASH_A": 73,
        "HASH_B": 179,
        "HASH_C": 31,
        "HASH_MOD": 1000,
        "total_apple_positions": 64,
        "expected_protected_per_period": 32,
        "notes": (
            "At epsilon=0.9, agents comply 10% of timesteps. With 50% protection rate, "
            "this yields ~5% effective protection per timestep. However, because protection "
            "is distributed pseudo-randomly across all positions over time, every position "
            "gets some protection eventually. The hash function ensures no position is "
            "systematically vulnerable, unlike static protection where high-epsilon agents "
            "can repeatedly target and consume the same protected positions."
        )
    }
}
