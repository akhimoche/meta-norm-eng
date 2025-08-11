## Fixing failure of runs 
import sys
from pathlib import Path

# Add repo root to sys.path so imports like 'from agents...' work
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.base_agent import ACTION_MAP
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from meltingpot import substrate
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for live updates on macOS

# ☑️ Standardized package imports
from env.mp_llm_env import LLMPrepObject
from utils.proc_funcs import get_changes_diff  # used for calibration
from utils.operator_funcs import get_north     # used for calibration

from agents.random_agent import RandomAgent
from agents.selfish import SelfishAgent

# === Config ===
env_name      = "commons_harvest__open"
num_players   = 5
window_size   = 50
interactive   = True  # True/False

# === Env setup ===
def create_env():
    roles = tuple(["default"] * num_players)
    return substrate.build(env_name, roles=roles)

env = create_env()
action_spec = env.action_spec()[0]
a_min, a_max = action_spec.minimum, action_spec.maximum

# Set up the sprite-to-symbol converter
ENV_NAME = "commons_harvest__open"
LABELS_DIR = ROOT_DIR / "utils" / "sprite_labels" / ENV_NAME
converter = LLMPrepObject(str(LABELS_DIR))

# === Initialize an observation for calibration ===
obs = env.reset()   # Must reset before calibration so `obs` exists

# === Calibration: discover each agent’s colour & rotate all to north ===
run_sprite_identification = True
if run_sprite_identification:
    colour_dict     = {}
    orientation_dict= {}
    position_dict   = {}

    # One-by-one, turn each agent left and see which sprite moved
    for t in range(num_players):
        test_actions = np.zeros(num_players, dtype=int)
        test_actions[t] = ACTION_MAP["TURN_LEFT"]  # integer ID for TURN_LEFT

        pre  = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        obs  = env.step(test_actions)
        post = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        changes = get_changes_diff(pre, post)

        sprite_key = next(k for k in changes if k.startswith("p_"))
        colour, orient = sprite_key.split("_")[1:3]
        colour_dict[t]      = colour
        orientation_dict[t] = orient
        position_dict[t]    = changes[sprite_key][0]

    # Rotate everyone north
    north_done = False
    while not north_done:
        rotation = np.zeros(num_players, dtype=int)
        for t in range(num_players):
            act_token, new_orient = get_north(orientation_dict[t])  # returns string token
            rotation[t] = ACTION_MAP[act_token]  # map token to int ID
            orientation_dict[t] = new_orient

        if rotation.sum() == 0:
            north_done = True
        else:
            obs = env.step(rotation)

# === Agent selection & instantiation ===
agent_types = [
    SelfishAgent,     # agent 0
    SelfishAgent,     # agent 1
    SelfishAgent,     # agent 2
    SelfishAgent,     # agent 3
    SelfishAgent      # agent 4
]

agents = []
for i, AgentClass in enumerate(agent_types):
    if AgentClass is SelfishAgent:
        agents.append(AgentClass(i, a_min, a_max, converter, colour_dict[i]))
    else:
        agents.append(AgentClass(i, a_min, a_max))

# === Tracking ===
social_welfare = []
done = False
t = 0

# === Optional animation ===
if interactive:
    plt.ion()
    fig_anim, ax_anim = plt.subplots()

# === Main loop ===
while not done and t < window_size:
    actions = [agent.act(obs) for agent in agents]
    timestep = env.step(actions)
    obs = timestep

    if interactive:
        frame = timestep.observation[0]["WORLD.RGB"]
        ax_anim.imshow(frame)
        fig_anim.canvas.draw()
        fig_anim.canvas.flush_events()
        plt.pause(0.2)
        ax_anim.clear()

    total = sum(timestep.reward).item()
    social_welfare.append(total)

    done = timestep.last()
    t += 1

# === Final plot ===
plt.ioff()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(social_welfare)
ax.set(
    xlabel="Timestep",
    ylabel="Total Social Welfare",
    title="Agent-based Run",
)
ax.grid(True)
plt.show()


