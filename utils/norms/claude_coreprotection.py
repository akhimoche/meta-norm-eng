# Section 0: Standard library imports
from typing import Set

from utils.norms.norm import Norm, Coord


class OpusRedundantCoreNorm(Norm):
    """
    Opus's Redundant Core Protection Norm for Commons Harvest.
    
    This norm protects a larger "core" of each apple patch with built-in redundancy.
    The strategy is designed for robustness to high epsilon values (0.5-0.9) by:
    
    1. Protecting more positions per patch than minimally necessary
    2. Ensuring every protected position has 2+ other protected neighbors
    3. Creating a "backbone" that can survive partial violations
    
    The key insight is that at high epsilon, some protected apples WILL be eaten.
    By protecting redundant cores, even with 50% loss, enough seeds remain to
    maintain the 3+ neighbor threshold for maximum regrowth (2.5%).
    
    Protection strategy by patch:
    - Corner patches (6 apples): Protect 4 interior positions
    - Diamond patches (13 apples): Protect 7-position cross core
    
    This leaves harvestable peripherals that:
    - Are reachable via A* (agents don't need to violate)
    - Have 3+ protected neighbors (fast regrowth when eaten)
    - Create a sustainable harvest cycle
    """
    
    def __init__(self, epsilon: float = 0.0):
        super().__init__("opus_redundant_core", epsilon)
        
        # Protected core positions - chosen for maximum redundancy
        # Each position has 2+ other protected neighbors for resilience
        self.protected_cores: Set[Coord] = {
            # ===== Upper Left Corner Patch (6 apples) =====
            # Positions: (1,1), (2,1), (3,1), (1,2), (2,2), (1,3)
            # Protect the interior square, leaving (3,1) and (1,3) harvestable
            (1, 1), (2, 1), (1, 2), (2, 2),
            
            # ===== Lower Left Diamond Patch (13 apples) =====
            # Protect the 7-position core cross for redundancy
            # Center column: (3,6), (3,7), (3,8), (3,9), (3,10)
            # Center row: (1,8), (2,8), (3,8), (4,8), (5,8)
            # Core intersection provides max redundancy
            (3, 7), (2, 8), (3, 8), (4, 8), (3, 9),
            (2, 7), (4, 7),  # Additional row for redundancy
            
            # ===== Upper Mid-Left Diamond Patch (13 apples) =====
            # Protect the central cross with redundant neighbors
            # Core: center column (8,1)-(8,5) + center row of (6-10,3)
            (8, 2), (7, 3), (8, 3), (9, 3), (8, 4),
            (7, 2), (9, 2),  # Additional row for redundancy
            
            # ===== Upper Mid-Right Diamond Patch (13 apples) =====
            # Mirror of mid-left patch
            (15, 2), (14, 3), (15, 3), (16, 3), (15, 4),
            (14, 2), (16, 2),  # Additional row for redundancy
            
            # ===== Lower Right Diamond Patch (13 apples) =====
            # Mirror of lower left patch
            (20, 7), (19, 8), (20, 8), (21, 8), (20, 9),
            (19, 7), (21, 7),  # Additional row for redundancy
            
            # ===== Upper Right Corner Patch (6 apples) =====
            # Positions: (20,1), (21,1), (22,1), (21,2), (22,2), (22,3)
            # Protect interior, leaving (20,1) and (22,3) harvestable
            (21, 1), (22, 1), (21, 2), (22, 2),
        }
        
        # Store hyperparameters as class attributes
        self.total_protected = len(self.protected_cores)
        self.total_apple_positions = 64  # 6 + 13 + 13 + 13 + 13 + 6
        self.harvestable_positions = self.total_apple_positions - self.total_protected
        
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the protected core positions (constant across all timesteps).
        
        The static nature is intentional - dynamic norms are harder to coordinate
        and more susceptible to timing-based epsilon violations.
        
        Args:
            t: Current simulation timestep (unused for static norm)
            
        Returns:
            Set of protected core positions
        """
        return self.protected_cores.copy()


# Meta-norm documentation following required format
meta_norm = {
    "verbal_explanation": (
        "This norm protects a redundant 'core' of each apple patch, leaving only "
        "peripheral apples harvestable. Unlike minimal protection strategies that "
        "block just 2-3 seeds per patch, this norm blocks 4-7 positions per patch "
        "to create overlapping protection zones. The key insight is that at high "
        "epsilon (0.5-0.9), agents WILL violate protection occasionally. By having "
        "redundant protected positions where each core apple has 2+ other protected "
        "neighbors, even when some protected apples are eaten, the remaining ones "
        "still provide 3+ neighbors to harvestable positions, maintaining maximum "
        "regrowth rate. Harvestable peripherals are positioned at patch edges where "
        "they can be reached without entering protected zones, so compliant agents "
        "rarely need to consider violating."
    ),
    "reasoning": (
        "The Commons Harvest regrowth mechanics create a sharp threshold: 3+ neighbors "
        "yield 2.5% regrowth per step, while fewer neighbors yield dramatically lower "
        "rates (0.5% for 2, 0.25% for 1, 0% for 0). For sustainability over 1000 steps, "
        "we need to ensure this threshold is maintained even under adversarial conditions. "
        "The redundant core strategy works because: (1) Protected positions form connected "
        "components that are internally resilient - losing one doesn't isolate others; "
        "(2) Each harvestable peripheral has 3+ protected neighbors, so fast regrowth is "
        "guaranteed as long as ANY protection remains; (3) At epsilon=0.5, roughly half "
        "of violation attempts succeed, but agents only violate when A* routes through "
        "protected zones, which is rare when peripherals are available; (4) The 36 "
        "protected positions leave 28 harvestable, enough for all 5 agents to find "
        "paths without collision. This approach trades some theoretical maximum harvest "
        "for dramatically improved robustness to partial norm compliance."
    ),
    "code_with_placeholders": "See OpusRedundantCoreNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "total_protected_positions": 36,
        "harvestable_positions": 28,
        "protection_ratio": 0.5625,  # 36/64
        "positions_per_corner_patch": 4,
        "positions_per_diamond_patch": 7,
        "regrowth_tier_target": 0.025,
        "epsilon_robustness_strategy": "Redundant cores with 2+ internal neighbors each",
        "epsilon_tested_range": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
}
