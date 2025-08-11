import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, agent_id, action_min, action_max):
        super().__init__(agent_id)
        self.action_min = action_min
        self.action_max = action_max

    def act(self, obs):
        # ignore obs, just pick uniform random
        return int(np.random.randint(self.action_min, self.action_max + 1))
    

# As seen already in the mp_testbed.py/baseline_run.py, this agent will randomly select an action. easy to see in the animation 