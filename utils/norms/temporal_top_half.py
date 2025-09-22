# Section 0: Standard library imports
from typing import Set, Tuple
from .norm import Norm, Coord

# Section 1: TemporalTopHalf norm
class TemporalTopHalf(Norm):
    """
    A norm that blocks the top half of the tree area for the first N timesteps.
    
    Based on the commons_harvest__open map:
    - Had some trouble getting a one to one map to grid conversion that is relaible. 
    - Best lead so far: 144 rows x 192 columns
    
    Example usage:
        # Block top half for first 25 timesteps (parameters are hardcoded in the norm)
        norm = TemporalTopHalf(epsilon)
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize a temporal top half blocker norm.
        
        Args:
            epsilon: Probability of ignoring this norm (0.0 = always obey, 1.0 = always ignore)
        """
        super().__init__("temporal_top_half", epsilon)
        
        # Timestep dictionary approach 
        # {timestep: blocked_positions}
        self.timestep_rules = {
            0: self._get_top_half_positions(),    # Timesteps 0-25: block top half
            26: set()                             # Timesteps 26+: block nothing
        }
    
    def _get_top_half_positions(self):
        """Helper to get top half positions."""
        positions = set()
        # Block the top half: rows 0-71 across full width cols 0-191
        for row in range(0, 72):  # rows 0-71 (top half of 144 rows)
            for col in range(0, 192):  # cols 0-191 (full width)
                positions.add((row, col))
        return positions
    
    def get_blocked_positions(self, t: int) -> Set[Coord]:  # t = timestep
        """
        Return blocked positions for this timestep.
        
        Args:
            t: Current simulation timestep (0-based)
            
        Returns:
            Set of blocked positions for this timestep
        """
        if t <= 25:  # First 25 timesteps: block top half
            return self.timestep_rules[0].copy()
        else:  # After timestep 25: block nothing
            return set()
    
# Looks a bit redundnant tbf, should retunrn to it. 