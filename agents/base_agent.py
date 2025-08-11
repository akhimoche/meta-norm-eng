import numpy as np

ACTION_MAP = {
    "NOOP": 0,
    "FORWARD": 1, "BACKWARD": 2,
    "STEP_LEFT": 3, "STEP_RIGHT": 4,
    "TURN_LEFT": 5, "TURN_RIGHT": 6,
    "FIRE_ZAP": 7,
}

class BaseAgent:
    """
    Base class for all agents.
    """
    def __init__(self, agent_id, seed: int | None = None, action_map: dict | None = None):
        self.id = int(agent_id)
        # per‑instance RNG prevents synchronized choices
        self.rng = np.random.default_rng(None if seed is None else seed + self.id)
        # single source of truth for action IDs
        self.action_map = action_map or ACTION_MAP

    def act(self, obs):
        raise NotImplementedError
