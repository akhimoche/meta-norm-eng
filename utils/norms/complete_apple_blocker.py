# Section 0: Standard library imports
from typing import Set
from utils.norms.norm import Norm, Coord

# Section 1: CompleteAppleBlocker norm
class CompleteAppleBlocker(Norm):
    """
    A debugging norm that blocks all apple positions to confirm grid system and coordinates.
    
    This norm was used as a debugging tool to confirm the grid coordinate system
    used by the environment and to verify the exact positions of all apples.
    It blocks all 64 apple coordinates from the ASCII map to test coordinate accuracy.
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize the complete apple blocker norm.
        
        Args:
            epsilon: Probability of ignoring this norm (0.0 = always obey, 1.0 = always ignore)
        """
        super().__init__("complete_apple_blocker", epsilon)
        
        # Debugging tool: Block all apple coordinates to confirm grid system
        # Uses ASCII map coordinates (18 rows × 24 columns) to block all 64 apples
        self.blocked_apple_positions = set()
        
        # Block all apple coordinates from ASCII map (18 rows × 24 columns)
        self.blocked_apple_positions = {
            # Upper left corner
            (1,1), (2,1), (3,1), (1,2), (2,2), (1,3),
            
            # Lower left
            (3, 6), (2, 7), (3, 7), (4, 7), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), 
            (2, 9), (3, 9), (4, 9), (3, 10),
            
            # Upper left
            (8, 1), (7, 2), (8, 2), (9, 2), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), 
            (7, 4), (8, 4), (9, 4), (8, 5),
            
            # Upper right
            (15, 1), (14, 2), (15, 2), (16, 2), (13, 3), (14, 3), (15, 3), (16, 3), (17, 3), 
            (14, 4), (15, 4), (16, 4), (15, 5),
            
            # Lower right
            (20, 6), (19, 7), (20, 7), (21, 7), (18, 8), (19, 8), (20, 8), (21, 8), (22, 8), 
            (19, 9), (20, 9), (21, 9), (20, 10),
            
            # Upper right corner
            (20, 1), (21, 1), (22, 1), (21, 2), (22, 2), (22, 3)
        }
    
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the blocked apple positions.
        
        Args:
            t: Current simulation timestep
            
        Returns:
            Set of blocked apple positions
        """
        return self.blocked_apple_positions.copy()
    
    def add_positions(self, positions: Set[Coord]):
        """Add more positions to block (for debugging/testing)"""
        self.blocked_apple_positions.update(positions)
    
    def clear_positions(self):
        """Clear all blocked positions (for debugging/testing)"""
        self.blocked_apple_positions.clear()
