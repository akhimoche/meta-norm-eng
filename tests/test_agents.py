# tests/test_agents.py

import sys, os
# 1) Compute project root and Phase2 folder paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PHASE2_ROOT  = os.path.join(PROJECT_ROOT, 'Phase2')

# 2) Add Phase2 (so Python can find Phase2/agents) to sys.path
sys.path.insert(0, PHASE2_ROOT)

import pytest
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent

class DummyObs:
    pass

def test_base_agent_act_not_implemented():
    agent = BaseAgent(agent_id=0)
    with pytest.raises(NotImplementedError):
        agent.act(DummyObs())

def test_random_agent_returns_valid_action():
    action_min, action_max = 0, 7
    agent = RandomAgent(agent_id=0, action_min=action_min, action_max=action_max)
    action = agent.act(DummyObs())
    assert isinstance(action, int)
    assert action_min <= action <= action_max
