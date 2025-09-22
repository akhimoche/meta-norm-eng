# Section 0: Standard library imports
from typing import Tuple

# Section 1: Orientation helper
def get_north(orientation: str) -> Tuple[str, str]:
    """
    Rotate the agent one step toward 'north'.
    
    Args:
        orientation: Current orientation ("north", "east", "south", "west")
        
    Returns:
        (action_token, new_orientation): Action to take and resulting orientation
        
    Notes:
        - This is incremental; you may need multiple calls across timesteps
          to fully rotate depending on current orientation.
    """
    if orientation == "north":
        return "NOOP", "north"
    elif orientation == "east":
        return "TURN_LEFT", "north"
    elif orientation == "south":
        return "TURN_LEFT", "east"  # Two steps needed
    elif orientation == "west":
        return "TURN_RIGHT", "north"
    else:
        return "NOOP", orientation  # Unknown orientation, do nothing

# Section 2: Movement helper
def move_agent(coord_init: Tuple[int, int], coord_final: Tuple[int, int]) -> str:
    """
    Given two adjacent grid cells, return the movement token to go init → final.
    
    Args:
        coord_init: Starting position (row, col)
        coord_final: Target position (row, col)
        
    Returns:
        Action token: "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", or "NOOP"
        
    Notes:
        - FORWARD: row-1 (north), BACKWARD: row+1 (south)
        - STEP_LEFT: col-1 (west), STEP_RIGHT: col+1 (east)
        - Returns "NOOP" if not a valid 4-connected step
    """
    r0, c0 = coord_init
    r1, c1 = coord_final

    if (r0, c0) == (r1, c1):
        return "NOOP"

    # Movement logic matching original working code:
    # dc = c1 - c0, dr = r1 - r0
    dc = c1 - c0
    dr = r1 - r0
    
    if dc == -1:
        return "FORWARD"    # col decreases (north)
    elif dc == 1:
        return "BACKWARD"   # col increases (south)
    elif dr == -1:
        return "STEP_LEFT"  # row decreases (west)
    elif dr == 1:
        return "STEP_RIGHT" # row increases (east)

    return "NOOP"  # not a valid 4-connected step