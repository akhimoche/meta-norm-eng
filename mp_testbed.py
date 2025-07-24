from meltingpot import substrate
from matplotlib import pyplot as plt
from env.mp_llm_env import LLMPrepObject
import numpy as np
import utils.proc_funcs
import utils.operator_funcs
# Turn on interactive mode so figures update live during the loop
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


# Define the substrate (environment) name
env_name = 'commons_harvest__open'  # example environment from Melting Pot
num_players=5
roles = tuple(['default' for _ in range(num_players)]) #Default role for agents in MP, maybe we can change this for our extnesion project? 
# Build the environment using the substrate's configuration
env = substrate.build(env_name, roles=roles)
converter = LLMPrepObject(f'/Users/michaeltaliotis/Desktop/MAL/meta-ssd/sprite_labels/{env_name}')
timestep_dictionary = {"Timestep 0": {}}

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.maximum
action_min = action_spec.minimum

#Exploring the action space
#print("num_actions:", num_actions)
#print("min:", action_min, "| max:", action_max)
#8 actions, move x4 directions, rotate x2 directions, NOOP (do nothing), FIRE_ZAP=big yellow shape that flashes sometimes =>peer punishment?

# reset environment
obs = env.reset()

# NOTE: The original variable name 'sprites_already_found' was a bit confusing.
# It seems this block actually *generates* sprite labels and rotates all players to face north.
# Renaming it for clarity, though I might be missing some context.
# Set this to True if you want to run the identification and alignment steps.
run_sprite_identification = True

if run_sprite_identification is True:
    # ---- Get player characteristics ----- #
    colour_dict = {}
    orientation_dict = {}
    position_dict = {}

    for t in range(num_players):
        # turn agent associated with player index t, all other players NOOP
        test_actions = np.zeros((num_players), dtype=int) #action list where all players do nothing
        test_actions[t] = 5 #except for player t, who turns left

        # Get dictionary of states before and after turn movement
        # Get the current PIXEL image of the envt before anyone moves
        current_screen_frame = obs.observation[0]['WORLD.RGB']
        # Converts image to a symbolic dictionary of entities on the grid (eg "p_blue_north": [(7.7)])<- to feed into LLM?
        dict_of_states_pre = converter.image_to_state(current_screen_frame)['global'] 
        obs = env.step(test_actions) #actually run the envt for one timestep 
        next_screen_frame = obs.observation[0]['WORLD.RGB'] 
        dict_of_states_post = converter.image_to_state(next_screen_frame)['global'] #dictionary conversion AFTER action <- again for LLM?
        changes = utils.proc_funcs.get_changes_diff(dict_of_states_pre, dict_of_states_post) 
        #compare before vs after to see what changed? should just be the sprite for player t, rotating left  

        # Pulling from the changes dictionary, get all entities which start with "p" (i.e the player sprite that moved, t)
        player_sprite = {key: value for key, value in changes.items() if key.startswith("p")}
        single_key = next(iter(player_sprite)) # e.g p_blue_south, the string name of the CHNAGED sprite 
        #Split the name of the sprite into its components (player, colour, orientation, and store them)
        split_key = single_key.split("_") # e.g [p, blue, south]
        colour_dict[t] = split_key[1]  # e.g blue
        orientation_dict[t] = split_key[2] # e.g south
        position_dict[t] = player_sprite.get(single_key) #Save the grid coordinates (x, y) of the player sprite.
    # ---- Get player characteristics ----- #

    # ---- Rotate all players north  ----- #
    north_found = False
    while not north_found:
        rotation_action = np.zeros((num_players),dtype=int)
        for t in range(num_players):
            orientation = orientation_dict[t]
            player_action, new_orientation = utils.operator_funcs.get_north(orientation)
            rotation_action[t]=player_action
            # update orientation
            orientation_dict[t] = new_orientation

        if np.array_equal(rotation_action,np.zeros((num_players),dtype=int)):
            north_found = True
            break

        obs = env.step(rotation_action)
    # ---- Rotate all players north  ----- #

# print(colour_dict)
# print(orientation_dict)
# print(position_dict)

# Set up the variables that will drive the main simulation loop
done = False
t=0
window_size = 50 # Number of timesteps to run the simulation
while not done:
    # Example: random actions for each agent.
    # NOTE! This will BE random, it's not meant to look like the final strategy.
    # It's meant to be a placeholder, later to be replaced with an actual policy (eg from an LLM/trained agent)
    actions = np.random.randint(action_min,action_max+1, num_players)
    # Step through the environment, advance it by one timestep
    # and get the new observation, reward, and done status.
    timestep = env.step(actions)

    # Get world RGB (raw piixel form)from 0th agent (assuming full observability)
    # I thought we were mostly going to be working with partial observabillity? 
    screen_frame = timestep.observation[0]['WORLD.RGB']

    # Get dictionary of states from world RGB frame (for LLM?)
    processed = converter.image_to_state(screen_frame)['global']


    # Option 1 (Static View):
    # Uncomment the lines below to display a single frame at a time. (& comment out Option 2)
    # This is useful for inspecting a single observation manually.
    # Note: plt.show() is blocking — you must close the window to continue.
    
    # plt.imshow(screen_frame)
    # plt.show()

    #Option 2 (Smooth Animation - Recommended):
    plt.imshow(screen_frame)   # Display the current environment state
    plt.pause(0.3)             # Wait for 300 milliseconds so you can see it
    plt.clf()                  # Clear the figure before drawing the next one

    # total system reward is social welfare (summing up all individual rewards, eg from picking apples)
    processed["social_welfare"] = [sum(timestep.reward).item()] # turn into list for iteration in get_dynamic info

    #print(utils.proc_funcs.get_dynamic_info(processed))
    timestep_dictionary[f"Timestep {t+1}"] = processed # Store the processed timestep in the dictionary

    # Check if episode is done
    done = timestep.last()
    t+=1

    if t==window_size: # Stop after 50 timesteps if the game hasn't already ended "naturally", eg apples depleted?

        break

# Analyze and print state changes (e.g., sprite positions, social welfare) across all timesteps to summarize environment dynamics.
result = utils.proc_funcs.dynamic_process(timestep_dictionary)
print(f"\n Dynamic Process after {window_size} steps \n")

for i in range(len(result)):
     print(f"t{i+1}:", end="")
     print(result[f"t_{i+1}"])
