# Section 0: Standard library imports
from typing import Set
from utils.norms.norm import Norm, Coord

# Section 1: CompleteAppleBlocker norm
class CompleteAppleBlocker(Norm):
    """
    A test norm that blocks the top-left apple patch.
    
    This norm is designed to test whether we can block the top-left apple patch
    by blocking specific coordinates. We'll start with a single coordinate and
    build up to see what works.
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize the top-left blocker norm.
        
        Args:
            epsilon: Probability of ignoring this norm (0.0 = always obey, 1.0 = always ignore)
        """
        super().__init__("complete_apple_blocker", epsilon)
        
        # Manual testing approach - block all of row 1
        # Grid is 144 rows × 192 columns, so row 1 has columns 0-191
        # This will test if blocking the entire row affects apple behavior
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
        """Add more positions to block (for testing)"""
        self.blocked_apple_positions.update(positions)
    
    def clear_positions(self):
        """Clear all blocked positions"""
        self.blocked_apple_positions.clear()
