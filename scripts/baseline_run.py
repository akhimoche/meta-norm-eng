# Note: This demonstrates a random agent baseline for the Melting Pot Commons Harvest scenario.
# Note that the random agent has been HARD-CODED to select a random action from the action space.
# For the more sophisticated design, look at run_agents.py 

from meltingpot import substrate
import matplotlib
matplotlib.use('TkAgg')   # set backend once here
import matplotlib.pyplot as plt  # for plotting social welfare over time
import numpy as np  # for generating random actions


# Environment creation 
env_name = "commons_harvest__open"   # Melting Pot scenario
num_players = 5                      # Number of agents

# Optional helper function to create the environment
def create_env(env_name=env_name, num_agents=num_players):
    """
    Creates and returns a Melting Pot environment with the given name and number of agents.
    Roles are all set to 'default'.
    """
    roles = tuple(["default"] * num_agents)
    return substrate.build(env_name, roles=roles)

# Build the environment
env = create_env(env_name, num_players)

# Placeholder: Role variation could be added later for experiments

# For Step 1, we skip LLM state conversion.
# We'll only track social welfare values per timestep in a list.
social_welfare_per_step = []

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.maximum
action_min = action_spec.minimum

# reset environment
obs = env.reset()  # starts the environment at timestep 0 and returns the first observation

# Placeholder: Orientation calibration (not needed for random baseline)
# In Phase 1, mp_testbed rotated all agents to face north so that movement decisions could be mapped consistently.
# For now we skip this because random actions don't depend on facing.
# If needed, insert calibration code here.

# Main simulation loop (random agents) 
done = False
t = 0
window_size = 50  # Number of timesteps to simulate

interactive_mode = False  # Toggle here: True = animation + welfare plot, False = welfare plot only

# If interactive mode, create figure for animation
if interactive_mode:
    plt.ion()  # enable interactive mode once at the start
    fig_anim, ax_anim = plt.subplots()
    fig_anim.show()

while not done:
    actions = np.random.randint(action_min, action_max + 1, num_players)
    timestep = env.step(actions)

    if interactive_mode:
        # Display the current frame of the environment
        screen_frame = timestep.observation[0]['WORLD.RGB']
        ax_anim.imshow(screen_frame)
        fig_anim.canvas.draw()
        fig_anim.canvas.flush_events()
        plt.pause(0.3)
        ax_anim.clear()

    # Calculate total social welfare (sum of all agent rewards)
    total_welfare = sum(timestep.reward).item()
    social_welfare_per_step.append(total_welfare)

    # Optional: Print for debugging
    # print(f"Step {t+1}: Total welfare = {total_welfare}")

    # --- Placeholders for removed Phase 1 features ---
    # Rendering (interactive mode) skipped:
    # In Phase 1, used plt.imshow + plt.pause to show env frames live.
    # For random-agent baseline, only plotting welfare at the end.
    # Could re-add for visual debugging later.
    
    # LLM state conversion skipped:
    # In Phase 1, used converter.image_to_state to get symbolic info.
    # Will add back in Step 3 for selfish/altruistic logic.

    # Check if episode has ended
    done = timestep.last()
    t += 1

    # Stop early if we reach the set window size
    if t == window_size:
        break

# --- Final welfare plot ---
plt.ioff()  # turn off interactive mode for any new figures

fig_welfare, ax_welfare = plt.subplots(figsize=(8, 6))  # width, height in inches
ax_welfare.plot(social_welfare_per_step)
ax_welfare.set_xlabel("Timestep")
ax_welfare.set_ylabel("Total Social Welfare")
ax_welfare.set_title("Random Agent Baseline - Commons Harvest Open")
ax_welfare.grid(True)
ax_welfare.set_aspect('auto')  # Ensure aspect ratio is not fixed

# NOTE: If interactive mode was run, this plot may appear squashed due to imshow's aspect settings.
# We can address that in a later step if needed.
plt.show()

# Placeholder: could also save the plot to file for later analysis
# plt.savefig("social_welfare_baseline.png")

