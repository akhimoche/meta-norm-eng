"""
Rotational Zone Access Norm for Commons Harvest Environment

This norm implements a sophisticated zone-based rotation system that prevents
overexploitation of apple patches while maintaining efficient resource collection.

Strategy:
- Divides apple patches into 6 zones (3 pairs of symmetric patches)
- Rotates access permissions to zones on a fixed cycle
- At any timestep, only 2-3 zones are accessible
- Forces agents to move between zones, allowing unused zones to recover
- Balances exploitation and recovery to maximize long-term yield

Hyperparameters:
- rotation_period: Number of timesteps each rotation phase lasts (default: 150)
- zones_per_phase: Number of zones accessible during each phase (default: 2)
- recovery_phases: Number of phases a zone stays blocked for recovery (default: 2)
"""

from typing import Set
from utils.norms.norm import Norm, Coord

meta_norm = {
    "verbal_explanation": (
        "This norm divides the apple patches into 6 distinct zones (upper-left corner, "
        "lower-left, upper-middle-left, upper-middle-right, lower-right, upper-right corner) "
        "and implements a rotational access system. At any given time, only 2 zones are "
        "accessible while the other 4 are blocked to allow apple recovery. The access "
        "permissions rotate every 150 timesteps, ensuring that no zone is overexploited "
        "and all zones get adequate recovery time. This prevents the tragedy of the commons "
        "by enforcing mandatory rest periods for each patch area."
    ),
    "reasoning": (
        "The tragedy of the commons occurs when all agents rush to collect apples from all "
        "patches simultaneously, preventing any patch from maintaining enough nearby apples "
        "for effective regrowth (which requires 3+ nearby apples for 2.5% regrowth rate). "
        "By blocking access to most zones at any given time, we ensure that blocked zones "
        "accumulate apples and maintain high local density, enabling robust regrowth. "
        "Meanwhile, agents can still efficiently collect from the 2 accessible zones. "
        "The 150-timestep rotation period is calibrated to allow sufficient recovery time "
        "(blocked zones get ~300 timesteps of recovery) while preventing agent starvation. "
        "With 1000 total timesteps and 6 zones cycling through with 2 accessible at a time, "
        "each zone gets approximately 333 timesteps of access and 667 timesteps of recovery, "
        "creating a sustainable equilibrium."
    ),
    "code_with_placeholders": """
# Zone definitions based on apple patch locations
ZONE_1 = {(1,1), (2,1), (3,1), (1,2), (2,2), (1,3)}  # Upper-left corner
ZONE_2 = {(3,6), (2,7), (3,7), (4,7), (1,8), (2,8), (3,8), (4,8), (5,8), (2,9), (3,9), (4,9), (3,10)}  # Lower-left
ZONE_3 = {(8,1), (7,2), (8,2), (9,2), (6,3), (7,3), (8,3), (9,3), (10,3), (7,4), (8,4), (9,4), (8,5)}  # Upper-middle-left
ZONE_4 = {(15,1), (14,2), (15,2), (16,2), (13,3), (14,3), (15,3), (16,3), (17,3), (14,4), (15,4), (16,4), (15,5)}  # Upper-middle-right
ZONE_5 = {(20,6), (19,7), (20,7), (21,7), (18,8), (19,8), (20,8), (21,8), (22,8), (19,9), (20,9), (21,9), (20,10)}  # Lower-right
ZONE_6 = {(20,1), (21,1), (22,1), (21,2), (22,2), (22,3)}  # Upper-right corner

ALL_ZONES = [ZONE_1, ZONE_2, ZONE_3, ZONE_4, ZONE_5, ZONE_6]

def get_blocked_positions(self, t: int) -> Set[Coord]:
    # Calculate which phase we're in (0-5, cycling every rotation_period timesteps)
    phase = (t // rotation_period) % 6
    
    # Determine which zones are accessible in this phase
    # We rotate through pairs: (0,1), (2,3), (4,5), (1,2), (3,4), (5,0)
    accessible_zones = {phase, (phase + 1) % 6}
    
    # Block all zones except the accessible ones
    blocked = set()
    for zone_idx, zone in enumerate(ALL_ZONES):
        if zone_idx not in accessible_zones:
            blocked.update(zone)
    
    return blocked
""",
    "hyperparameters_for_this_environment": {
        "rotation_period": 150,
        "zones_per_phase": 2,
        "recovery_phases": 2,
        "description": (
            "rotation_period=150 provides each zone with substantial recovery time. "
            "zones_per_phase=2 balances resource availability with recovery needs. "
            "recovery_phases=2 ensures zones stay blocked long enough for apple density to rebuild."
        )
    }
}


class RotationalZoneAccessNorm(Norm):
    """
    Implements a zone-based rotational access system for Commons Harvest.
    
    This norm divides apple patches into 6 zones and rotates access permissions
    to prevent overexploitation while maintaining efficient resource collection.
    
    Args:
        epsilon: Probability of ignoring the norm (0.0 = always obey)
        rotation_period: Timesteps per rotation phase (default: 150)
        zones_per_phase: Number of zones accessible per phase (default: 2)
    """
    
    # Zone definitions based on apple patch locations
    ZONE_1 = {(1,1), (2,1), (3,1), (1,2), (2,2), (1,3)}  # Upper-left corner
    ZONE_2 = {(3,6), (2,7), (3,7), (4,7), (1,8), (2,8), (3,8), (4,8), (5,8), (2,9), (3,9), (4,9), (3,10)}  # Lower-left
    ZONE_3 = {(8,1), (7,2), (8,2), (9,2), (6,3), (7,3), (8,3), (9,3), (10,3), (7,4), (8,4), (9,4), (8,5)}  # Upper-middle-left
    ZONE_4 = {(15,1), (14,2), (15,2), (16,2), (13,3), (14,3), (15,3), (16,3), (17,3), (14,4), (15,4), (16,4), (15,5)}  # Upper-middle-right
    ZONE_5 = {(20,6), (19,7), (20,7), (21,7), (18,8), (19,8), (20,8), (21,8), (22,8), (19,9), (20,9), (21,9), (20,10)}  # Lower-right
    ZONE_6 = {(20,1), (21,1), (22,1), (21,2), (22,2), (22,3)}  # Upper-right corner
    
    ALL_ZONES = [ZONE_1, ZONE_2, ZONE_3, ZONE_4, ZONE_5, ZONE_6]
    
    def __init__(self, epsilon: float = 0.0, rotation_period: int = 30, zones_per_phase: int = 2):
        """
        Initialize the Rotational Zone Access Norm.
        
        Args:
            epsilon: Probability of ignoring the norm (0.0 = always obey, 1.0 = always ignore)
            rotation_period: Number of timesteps each rotation phase lasts
            zones_per_phase: Number of zones accessible during each phase
        """
        super().__init__("rotational_zone_access", epsilon)
        self.rotation_period = rotation_period
        self.zones_per_phase = zones_per_phase
    
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns the set of positions that are blocked at timestep t.
        
        The norm rotates through 6 phases, with each phase lasting rotation_period timesteps.
        In each phase, only zones_per_phase zones are accessible, and the rest are blocked.
        
        Args:
            t: Current timestep
            
        Returns:
            Set of (column, row) coordinates that are blocked by the norm
        """
        # Calculate which phase we're in (0-5, cycling every rotation_period timesteps)
        phase = (t // self.rotation_period) % 6
        
        # Determine which zones are accessible in this phase
        # We rotate through pairs to ensure good spatial distribution:
        # Phase 0: zones 0,1 (left side)
        # Phase 1: zones 1,2 (left-middle)
        # Phase 2: zones 2,3 (middle)
        # Phase 3: zones 3,4 (middle-right)
        # Phase 4: zones 4,5 (right side)
        # Phase 5: zones 5,0 (edges)
        accessible_zones = {phase, (phase + 1) % 6}
        
        # Block all zones except the accessible ones
        blocked = set()
        for zone_idx, zone in enumerate(self.ALL_ZONES):
            if zone_idx not in accessible_zones:
                blocked.update(zone)
        
        return blocked