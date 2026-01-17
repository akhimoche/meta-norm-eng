# Section 0: Standard library imports
from typing import Set
from utils.norms.norm import Norm, Coord

# Section 1: SacredApples norm
class SacredApples(Norm):
    """
    A norm that permanently blocks specific apple positions for the entire simulation.
    
    This norm blocks certain apple positions to create "reserve" apples that maintain
    apple clusters and optimal regrowth patterns. Agents cannot move to these positions
    under any circumstances (hard blocking).
    
    Example usage:
        # Block specific apple positions permanently (positions are hardcoded in the norm)
        norm = SacredApples(epsilon=0.1)
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize a sacred apples norm.
        
        Args:
            epsilon: Probability of ignoring this norm (0.0 = always obey, 1.0 = always ignore)
            !Communicate better that the central epsilon knob is in the driver file run_agents.
            Maybe remove references to epsilon here? but they still take the valur form driver 
            file and run with it?  
        """
        super().__init__("sacred_apples", epsilon)
        
        # Hardcoded: block specific apple positions permanently
        self.blocked_apple_positions = {
            (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 8), (5, 8), (2, 7), 
            (8, 1), (8, 3), (9, 2), (9, 3), (15, 1), (15, 3), (14, 2), (14, 3), 
            (19, 8), (20, 8), (22, 8), (20, 7), (21, 8), (22, 1), (22, 2), (21, 1), (21, 2)
        }
    
    def get_blocked_positions(self, t: int) -> Set[Coord]:  # t = timestep
        """
        Return the blocked apple positions (same for all timesteps).
        
        Args:
            t: Current simulation timestep (ignored for static blocker)
            
        Returns:
            Set of blocked apple positions
        """
        return self.blocked_apple_positions.copy()
