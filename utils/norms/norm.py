# Section 0: Standard library imports. Don't strictly need those. 
from typing import Set, Tuple # Supposedly used commonly for error detection?
Coord = Tuple[int, int] # Improves readbility 

# Section 1: Base Norm class. 
class Norm: # How much explanation do we need for this? Don't want to bloat it too much. 
    """
    Base class for all norms.
    
    A norm is simply a provider of blocked positions at different timesteps.
    The agent combines these with physical obstacles (walls) for A* pathfinding.
    
    Architecture:
    - Agent gets physical obstacles from vision
    - Agent gets norm obstacles from norm.get_blocked_positions(t)
    - Agent combines them: all_obstacles = physical_obstacles | norm_obstacles
    - Epsilon compliance: if random.random() < epsilon, use only physical_obstacles
    """
    
    def __init__(self, name: str, epsilon: float = None):
        """
        Initialize a norm. (basic properties)
        
        Args:
            name: Name of the norm
            epsilon: Probability of ignoring this norm (0.0 = always obey, 1.0 = always ignore)
                    Must be explicitly provided - no default value allowed
        """
        if epsilon is None:
            raise ValueError(f"Epsilon must be explicitly provided when creating norm '{name}'. "
                           f"Please set epsilon in the driver file (run_agents.py) and pass it to the norm constructor.")
        
        self.name = name
        self.epsilon = max(0.0, min(1.0, float(epsilon)))  # Clamp to [0, 1]
    
    def get_blocked_positions(self, t: int) -> Set[Coord]:  # t = timestep
        """
        Return the set of blocked positions at the given timestep.
        
        This is the core interface that all norms must implement.
        The agent will call this method and combine the result with physical obstacles.
        
        Args:
            t: Current simulation timestep (0-based)
            
        Returns:
            Set of (row, col) coordinates that are blocked by this norm
        """
        raise NotImplementedError("Subclasses must implement get_blocked_positions")
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}', epsilon={self.epsilon})"
