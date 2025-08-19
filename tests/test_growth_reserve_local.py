# tests/test_growth_reserve_local.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.norms.growth_reserve_local import GrowthReserveLocal
from utils.operator_funcs import a_star_with_norms

def test_penalty_triggers_when_post_neighbors_below_K():
    # A tiny cluster: apples at (0,1) only. Harvesting it would leave 0 neighbors.
    norm = GrowthReserveLocal(radius=2.0, K=3, penalty=5.0, seed=0)
    norm.update_apples({(0, 1)})
    norm.set_epsilon("A", 0.0)

    # Moving into the apple should incur a penalty > 0
    p = norm.step_penalty("A", (0, 0), (0, 1))
    assert p > 0.0

def test_a_star_avoids_penalized_apple_when_detour_is_cheaper():
    # Grid 3x4, start (0,0) -> goal (0,3). Apple at (0,1) would violate K=3, penalty=5.
    # Direct path cost with penalty: 1 + (1+5) + 1 = 8
    # Detour path cost (down-right-right-up): 4
    norm = GrowthReserveLocal(radius=2.0, K=3, penalty=5.0, seed=0)
    apples = {(0, 1)}
    norm.update_apples(apples)
    norm.set_epsilon("A", 0.0)

    obstacles = set()
    grid_size = (3, 4)  # rows, cols

    path = a_star_with_norms(
        start=(0, 0), goal=(0, 3),
        obstacles=obstacles, grid_size=grid_size,
        agent_id="A",
        norms_active=True,
        norms_blocked=norm.is_blocked,
        norms_penalty=norm.step_penalty
    )
    # Path should detour and therefore NOT step on (0,1)
    assert (0, 1) not in path, f"Expected detour around penalized apple, got {path}"

def test_epsilon_one_ignores_penalty_and_takes_shortcut():
    norm = GrowthReserveLocal(radius=2.0, K=3, penalty=5.0, seed=0)
    norm.update_apples({(0, 1)})
    norm.set_epsilon("A", 1.0)  # always ignore soft penalty

    path = a_star_with_norms(
        start=(0, 0), goal=(0, 3),
        obstacles=set(), grid_size=(3, 4),
        agent_id="A",
        norms_active=True,
        norms_blocked=norm.is_blocked,
        norms_penalty=norm.step_penalty
    )
    # With penalty ignored, shortest path goes straight through (0,1)
    assert (0, 1) in path, f"Expected straight path through apple, got {path}"
