# Section 0: Standard library imports
import random
from typing import Dict, List, Tuple, Optional, Set

# Third-party imports
import numpy as np

# Project imports
from .base_agent import BaseAgent, ACTION_MAP
from utils.pathfinding import a_star
from utils.norms.norm import Norm
from utils.operator_funcs import move_agent


#Section 1: SelfishAgent class definition 
class SelfishAgent(BaseAgent):
    """
    Heuristic agent that finds the nearest apple using A* pathfinding.
    Integrates with norms by treating norm-blocked tiles as obstacles.
    """

    def __init__(self, agent_id, colour, converter, norm: Optional[Norm] = None):
        super().__init__(agent_id, colour, converter)
        self.norm = norm  # Optional norm for obstacle handling

    # Section 2: Main decision loop 
    def act(self, obs, t=0):  # t = timestep
        """
        Main decision loop:
        Complete description when finalised 
        """
        # Step 1: Get symbolic state from RGB observation
        state = self.converter.image_to_state(obs.observation[0]["WORLD.RGB"])["global"]

        # Step 2: Locate this agent's position by its color label
        my_positions = []
        for label, positions in state.items():
            if label.startswith(f"p_{self.colour}_"):
                my_positions.extend(positions)
        my_position = my_positions[0] if my_positions else None
        
        # Check if we found our position
        if my_position is None:
            return ACTION_MAP["NOOP"]
        
        # Step 3: Find all apples (case-insensitive to catch all variations)
        apples = []
        for label, positions in state.items():
            if "apple" in label.lower():  # .lower() handles "Apple", "APPLE", "apple_1", etc.
                apples.extend(positions)

        # Step 4: Find obstacles (walls + norm-blocked tiles)
        # Get physical obstacles from vision
        physical_obstacles = set()
        for label, positions in state.items():
            if "wall" in label.lower():
                physical_obstacles.update(positions)  # Add all wall positions to obstacle set
            # NOTE: Removed other agents as obstacles - they should be able to move through each other
        
        # Get norm obstacles if norm exists
        norm_obstacles = set()
        if self.norm is not None:
            norm_obstacles = self.norm.get_blocked_positions(t)
        
        # Combine obstacles
        all_obstacles = physical_obstacles | norm_obstacles
        
        # Handle epsilon disobedience
        if self.norm is not None and random.random() < self.norm.epsilon:
            # Disobey norm: only use physical obstacles
            all_obstacles = physical_obstacles

        # Step 5: Find the best apple by running A* for each one (mayeb too inefficient apporach? )
        # Initialize tracking variables - use infinity so any real path will be better
        best_path = None
        best_score = float('inf')
        
        # Test each apple to find the one with shortest path
        for apple in apples:
            # Run A* pathfinding from my position to this apple
            path = a_star(my_position, apple, all_obstacles)
            
            # Skip if no path exists (empty list)
            if not path:
                continue
            
            # Calculate path length (number of steps to reach apple)
            path_length = len(path) - 1  # -1 because path includes start position
            
            # Update best path if this one is shorter
            if path_length < best_score:
                best_score = path_length
                best_path = path

        # Step 7: No reachable apple → do nothing (wait for apples to regrow)
        if not best_path:
            return ACTION_MAP["NOOP"]

        # Step 8: Take the first step toward the apple
        action_name = move_agent(my_position, best_path[1])
        action_id = ACTION_MAP[action_name]
        return action_id
