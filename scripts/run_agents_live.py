# Section 0: Standard library imports
import sys
import argparse
import csv
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib
from collections import Counter
import matplotlib.pyplot as plt

# Add project root to Python path for imports (hack, should be resolved)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Project imports
from agents.base_agent import ACTION_MAP
from agents.selfish import SelfishAgent
from env.mp_llm_env import LLMPrepObject # this is probly overcomplicated, might remove later
from utils.proc_funcs import get_changes_diff
from utils.operator_funcs import get_north
from meltingpot import substrate

# Configure matplotlib backend for macOS compatibility
matplotlib.use("TkAgg")

# Section 1: Configuration (MOVE ALL TO RUN EXPERIMENT.Py?), or after creating env? 
# Also, suggestion to log these under a @dataclass? class ExperimentConfig: ... overkill? 
env_name = "commons_harvest__open"
num_players = 5
window_size = 200 # Agreed upon standard for exeperiments is 1000 timesteps. 
interactive = True  # Debug tool - compartmentalize later
save_data = False  # Set to True to save experiment data to data/ folder

# Agent configuration
agent_types = [SelfishAgent] * num_players  

# Norm configuration - SELECT YOUR NORM HERE
# Available norms: Use utils.norms.loader.print_available_norms() to see all options
# Options: See norm names, or "None" for baseline
norm_type = "sacred_apples"  
# Epsilon settings for norm compliance (0.0 = always obey, 1.0 = always ignore)
epsilon_all = 0.2 # Remember! 80% compliance is epsilon 0.2
epsilon_overrides = {}  # Per-agent overrides: {"0": 0.2, "3": 0.5}


# Norm setup - dynamically load the norm using auto-discovery
from utils.norms.loader import get_norm
norm = get_norm(norm_type, epsilon=epsilon_all)

# Section 2: Environment setup
roles = ["default"] * num_players
env = substrate.build(env_name, roles=roles)
action_spec = env.action_spec()[0]
a_min, a_max = action_spec.minimum, action_spec.maximum

# Initialize environment
obs = env.reset()

# Section 3: Agent Calibration

# Sprite identification: extract agent positions, colors, and orientations from RGB images
# (Required because MeltingPot only provides RGB observations, not direct position data)
labels_dir = Path(__file__).resolve().parents[1] / "utils" / "sprite_labels" / env_name
converter = LLMPrepObject(str(labels_dir))

colour_dict = {}
orientation_dict = {}
position_dict = {}

# Identify each agent by testing movements and detecting sprite changes
for agent_id in range(num_players):
    test_actions = np.zeros(num_players, dtype=int)
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
    rotation_actions = np.zeros(num_players, dtype=int)
    
    for agent_id in range(num_players):
        action_token, new_orientation = get_north(orientation_dict[agent_id])
        rotation_actions[agent_id] = ACTION_MAP[action_token]
        orientation_dict[agent_id] = new_orientation

    # If no rotations needed, we're done
    if rotation_actions.sum() == 0:
        rotation_complete = True
    else:
        obs = env.step(rotation_actions)

# Section 4: Agent Creation
# Create actual agent instances to control the MeltingPot player slots
agents = []
for i, AgentClass in enumerate(agent_types):
    if AgentClass is SelfishAgent:
        # SelfishAgent needs agent_id, colour, converter, and optional norm
        agents.append(AgentClass(i, colour_dict[i], converter, norm=norm))
    else:
        # Other agent types only need basic parameters
        # TODO: When implementing AltruisticAgent, consider if it needs converter/colour_dict
        agents.append(AgentClass(i, a_min, a_max))

# Section 6: Simulation Setup
social_welfare = []
done = False
t = 0


# Animation Setup (Optional), Compartmentalise as a debug tool later/visualisation.py in utils? 
if interactive:
    plt.ion()
    fig_anim, ax_anim = plt.subplots()

# Section 8: Main Simulation Loop
while not done and t < window_size:
    # Update norm timestep if it exists
    if norm is not None and hasattr(norm, 'update_timestep'):
        norm.update_timestep(t)
    
    # Compute current agent positions from RGB

    # Get actions from all agents and step the environment
    actions = [agent.act(obs, t) for agent in agents]
    timestep_data = env.step(actions)
    obs = timestep_data  # Update observations for next iteration

    # Animation (compartmentalize later)
    if interactive:
        frame = timestep_data.observation[0]["WORLD.RGB"]
        ax_anim.imshow(frame)
        fig_anim.canvas.draw()
        fig_anim.canvas.flush_events()
        plt.pause(0.05)
        ax_anim.clear()

    # Record social welfare (sum of all agent rewards)
    total_reward = sum(timestep_data.reward).item()
    social_welfare.append(total_reward)

    # Check if episode is done
    done = timestep_data.last()
    t += 1

# Section 9: Data Analysis and Visualization 
# TODO: Save data to NumPy arrays in data/ folder (add to .gitignore)

# Convert to NumPy array for analysis
social_welfare_array = np.array(social_welfare, dtype=float)
cumulative_welfare = np.cumsum(social_welfare_array)

# Plot cumulative social welfare
plt.ioff()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(cumulative_welfare)
ax.set(
    xlabel="Timestep",
    ylabel="Cumulative Social Welfare", 
    title="Cumulative Team Reward per Episode",
)
ax.grid(True)
plt.show()

# Plot instantaneous rewards (reward rate)
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(social_welfare_array)
ax2.set(
    xlabel="Timestep",
    ylabel="Reward per Step",
    title="Instantaneous Team Reward",
)
ax2.grid(True)
plt.show()

