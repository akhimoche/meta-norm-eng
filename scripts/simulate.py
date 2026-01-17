# Section 0: Standard library imports
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Optional

# Third-party imports
import numpy as np

# Add project root to Python path for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Project imports
from agents.base_agent import ACTION_MAP
from agents.selfish import SelfishAgent
from env.mp_llm_env import LLMPrepObject
from utils.proc_funcs import get_changes_diff
from utils.operator_funcs import get_north
from utils.norms.loader import get_norm
from meltingpot import substrate


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    env_name: str
    norm_type: str
    epsilon: float
    agent_type: str = "selfish"  # Options: "selfish" (more types coming soon)
    num_players: int = 5
    timesteps: int = 500
    seed: Optional[int] = None  # Optional: Set if you need reproducible randomness
                                 # If runs look too similar, use different seeds per run


def run_simulation(config: SimulationConfig) -> dict:
    """
    Run a single simulation with the given configuration.
    
    Args:
        config: SimulationConfig object containing all simulation parameters
        
    Returns:
        Dictionary containing:
            - "social_welfare": np.array of per-timestep total rewards
            - "cumulative_welfare": np.array of cumulative total rewards
            - "metadata": dict of configuration used
    """
    
    # Optional: Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Section 1: Load norm
    norm = get_norm(config.norm_type, epsilon=config.epsilon)
    
    # Section 2: Environment setup
    roles = ["default"] * config.num_players
    env = substrate.build(config.env_name, roles=roles)
    action_spec = env.action_spec()[0]
    a_min, a_max = action_spec.minimum, action_spec.maximum
    
    # Initialize environment
    obs = env.reset()
    
    # Section 3: Agent Calibration
    # Sprite identification: extract agent positions, colors, and orientations from RGB images
    labels_dir = ROOT_DIR / "utils" / "sprite_labels" / config.env_name
    converter = LLMPrepObject(str(labels_dir))
    
    colour_dict = {}
    orientation_dict = {}
    position_dict = {}
    
    # Identify each agent by testing movements and detecting sprite changes
    for agent_id in range(config.num_players):
        test_actions = np.zeros(config.num_players, dtype=int)
        test_actions[agent_id] = 5  # TURN_LEFT for this agent only
        
        # Capture state before and after movement
        pre_state = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        obs = env.step(test_actions)
        post_state = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        
        # Find which sprite moved (indicates this agent's identity)
        changes = get_changes_diff(pre_state, post_state)
        sprite_key = next(k for k in changes if k.startswith("p_"))
        
        # Extract agent properties from sprite key (format: "p_{color}_{orientation}")
        colour, orientation = sprite_key.split("_")[1:3]
        colour_dict[agent_id] = colour
        orientation_dict[agent_id] = orientation
        position_dict[agent_id] = changes[sprite_key][0]
    
    # Rotate all agents to face north (standardize orientation for consistent behavior)
    rotation_complete = False
    while not rotation_complete:
        rotation_actions = np.zeros(config.num_players, dtype=int)
        
        for agent_id in range(config.num_players):
            action_token, new_orientation = get_north(orientation_dict[agent_id])
            rotation_actions[agent_id] = ACTION_MAP[action_token]
            orientation_dict[agent_id] = new_orientation
        
        # If no rotations needed, we're done
        if rotation_actions.sum() == 0:
            rotation_complete = True
        else:
            obs = env.step(rotation_actions)
    
    # Section 4: Agent Creation
    # Map agent type string to agent class
    if config.agent_type == "selfish":
        AgentClass = SelfishAgent
    else:
        raise ValueError(f"Unknown agent type: '{config.agent_type}'. Available: 'selfish'")
    
    agents = []
    for i in range(config.num_players):
        if AgentClass is SelfishAgent:
            agents.append(AgentClass(i, colour_dict[i], converter, norm=norm))
        else:
            # Future: Handle other agent types (altruistic, etc.)
            agents.append(AgentClass(i, a_min, a_max))
    
    # Section 5: Simulation Loop (No visualization)
    social_welfare = []
    done = False
    t = 0

    
    while not done and t < config.timesteps:
        # Update norm timestep if it exists
        if norm is not None and hasattr(norm, 'update_timestep'):
            norm.update_timestep(t)
        
        # Compute current agent positions from RGB

        # Get actions from all agents and step the environment
        actions = [agent.act(obs, t) for agent in agents]
        timestep_data = env.step(actions)
        obs = timestep_data
        
        # Record social welfare (sum of all agent rewards)
        total_reward = sum(timestep_data.reward).item()
        social_welfare.append(total_reward)
        
        # Check if episode is done
        done = timestep_data.last()
        t += 1
    
    # Section 6: Return data (no plotting)
    social_welfare_array = np.array(social_welfare, dtype=float)
    cumulative_welfare = np.cumsum(social_welfare_array)
    
    return {
        "social_welfare": social_welfare_array,
        "cumulative_welfare": cumulative_welfare,
        "metadata": {
            "env_name": config.env_name,
            "norm_type": config.norm_type,
            "epsilon": config.epsilon,
            "agent_type": config.agent_type,
            "num_players": config.num_players,
            "timesteps": config.timesteps,
            "actual_steps": len(social_welfare),
            "seed": config.seed,
        }
    }


if __name__ == "__main__":
    # Test simulation with default config
    test_config = SimulationConfig(
        env_name="commons_harvest__closed",
        norm_type="None",
        epsilon=0.0,
        agent_type="selfish",
        num_players=5,
        timesteps=100  # Short test run
    )
    
    print("Running test simulation...")
    result = run_simulation(test_config)
    print(f"Completed {result['metadata']['actual_steps']} timesteps")
    print(f"Final cumulative welfare: {result['cumulative_welfare'][-1]:.2f}")

