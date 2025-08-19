# tests/test_selfish_agent.py
import sys
from pathlib import Path
import numpy as np

# --- Ensure 'agents/' is importable when running this file directly or via pytest ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.selfish import SelfishAgent
from agents.base_agent import ACTION_MAP

# --- Minimal fakes so we don't need the Melting Pot env ---

class FakeConverter:
    """Mimics LLMPrepObject.image_to_state()."""
    def __init__(self, state_dict):
        self._state = {"global": state_dict}
    def image_to_state(self, frame):
        return self._state

class FakeObs:
    """Provides obs.observation[0]['WORLD.RGB'] like the env."""
    def __init__(self, h=5, w=5):
        self.observation = [{"WORLD.RGB": np.zeros((h, w, 3), dtype=np.uint8)}]

def _make_agent(converter):
    return SelfishAgent(
        agent_id=0,
        action_min=0,
        action_max=max(ACTION_MAP.values()),
        converter=converter,
        color="red",
        seed=123,  # deterministic fallback if needed
    )

def test_selfish_moves_toward_nearest_apple():
    """
    Agent at (2,2), apple at (2,3) → dy=+1 ⇒ BACKWARD (south) == 2.
    """
    state = {
        "p_red_north": [(2, 2)],
        "apple": [(2, 3)],
        # no walls/trees here
    }
    agent = _make_agent(FakeConverter(state))
    obs = FakeObs()
    action = agent.act(obs)
    assert action == ACTION_MAP["BACKWARD"], f"Expected BACKWARD, got {action}"

def test_selfish_random_fallback_when_no_path():
    """
    Agent boxed in by walls; A* finds no path ⇒ random translation {1,2,3,4}.
    """
    agent_pos = (2, 2)
    walls = [(2, 1), (2, 3), (1, 2), (3, 2)]
    state = {
        "p_red_north": [agent_pos],
        "apple": [(0, 0)],  # unreachable due to walls
        "wall": walls,
    }
    agent = _make_agent(FakeConverter(state))
    obs = FakeObs()
    action = agent.act(obs)
    assert action in {
        ACTION_MAP["FORWARD"], ACTION_MAP["BACKWARD"],
        ACTION_MAP["STEP_LEFT"], ACTION_MAP["STEP_RIGHT"],
    }, f"Expected a translation action, got {action}"
