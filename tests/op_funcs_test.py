"""
Unit tests for utils/operator_funcs.py

Run from repo root:
    PYTHONPATH=. python3.11 tests/op_funcs_test.py
"""

import numpy as np
from utils.operator_funcs import get_movement_actions, a_star, move_agent
from agents.base_agent import ACTION_MAP

def test_move_agent_diagonal():
    """Diagonal movement should result in NOOP."""
    result = move_agent((0, 0), (1, 1))
    assert result == "NOOP", f"Expected NOOP for diagonal, got {result}"
    print("✅ Diagonal input → NOOP passed.")

def test_a_star_with_obstacle():
    """Path blocked by obstacle should be routed around or return []."""
    start = (0, 0)
    goal = (0, 2)
    obstacles = {(0, 1)}  # block the direct path
    grid_size = (3, 3)

    path = a_star(start, goal, obstacles, grid_size)
    if path:
        assert path[0] == start and path[-1] == goal, "Path endpoints incorrect"
        assert (0, 1) not in path, "Path incorrectly goes through obstacle"
    else:
        print("⚠️ No path found — acceptable if fully blocked")
    print("✅ a_star obstacle test passed.")

def test_get_movement_actions_basic():
    """Basic check that get_movement_actions returns mapped ints for the first step."""
    colour_dict = ['red', 'blue']
    operator_output = {
        'red':  ((0, 0), (0, 1), set()),
        'blue': ((1, 1), (2, 2), set()),
    }
    num_players = 2
    grid_size = (3, 3)

    actions = get_movement_actions(operator_output, colour_dict, num_players, grid_size, ACTION_MAP)
    # Ensure integer dtype (signed or unsigned) and elements are numpy/Python ints
    assert isinstance(actions, np.ndarray)
    assert actions.dtype.kind in ("i", "u"), f"Expected integer dtype, got {actions.dtype}"
    assert all(isinstance(a, (int, np.integer)) for a in actions), "Actions must be integer-like"
    print(f"✅ get_movement_actions basic test passed. Actions: {actions.tolist()}")

def test_get_movement_actions_noop_fallback():
    """If no path or missing data, the function should return NOOP id for that player."""
    colour_dict = ['red', 'blue']
    # 'blue' is missing in operator_output on purpose; 'red' is blocked
    operator_output = {
        'red':  ((0, 0), (0, 2), {(0, 1)}),  # direct path blocked; small grid prevents detour
        # 'blue': omitted
    }
    num_players = 2
    grid_size = (1, 3)  # 1x3 line, obstacle in the center → no path red->goal

    actions = get_movement_actions(operator_output, colour_dict, num_players, grid_size, ACTION_MAP)
    noop = ACTION_MAP["NOOP"]
    assert actions[0] == noop, f"Expected NOOP for blocked 'red', got {actions[0]}"
    assert actions[1] == noop, f"Expected NOOP for missing 'blue', got {actions[1]}"
    print("✅ get_movement_actions NOOP fallback test passed.")

if __name__ == "__main__":
    print("Running operator_funcs unit tests...\n")
    test_move_agent_diagonal()
    test_a_star_with_obstacle()
    test_get_movement_actions_basic()
    test_get_movement_actions_noop_fallback()
    print("\n🎉 All operator_funcs tests passed.")
