from meltingpot import substrate
from matplotlib import pyplot as plt
from env.mp_llm_env import LLMPrepObject
import numpy as np
import utils.proc_funcs
import utils.operator_funcs


# Define the substrate (environment) name
env_name = 'commons_harvest__open'  # example environment from Melting Pot
num_players=5
roles = tuple(['default' for _ in range(num_players)])
# Build the environment using the substrate's configuration
env = substrate.build(env_name, roles=roles)
converter = LLMPrepObject(f'/home/akhimoche/meta-ssd/sprite_labels/{env_name}')
timestep_dictionary = {"Timestep 0": {}}

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.maximum
action_min = action_spec.minimum

# reset environment
obs = env.reset()

# set this to false to generate sprite labels
sprites_already_found = True

if sprites_already_found is True:
    # ---- Get player characteristics ----- #
    colour_dict = {}
    orientation_dict = {}
    position_dict = {}

    for t in range(num_players):
        # turn agent associated with player index t, all other players NOOP
        test_actions = np.zeros((num_players), dtype=int)
        test_actions[t] = 5 # turn left

        # Get dictionary of states before and after turn movement
        current_screen_frame = obs.observation[0]['WORLD.RGB']
        dict_of_states_pre = converter.image_to_state(current_screen_frame)['global']
        obs = env.step(test_actions)
        next_screen_frame = obs.observation[0]['WORLD.RGB']
        dict_of_states_post = converter.image_to_state(next_screen_frame)['global']
        changes = utils.proc_funcs.get_changes_diff(dict_of_states_pre, dict_of_states_post)

        # # get all keys which start with "p" (i.e all player sprites)
        player_sprite = {key: value for key, value in changes.items() if key.startswith("p")}
        single_key = next(iter(player_sprite)) # e.g p_blue_south
        split_key = single_key.split("_") # e.g [p, blue, south]
        colour_dict[t] = split_key[1]  # e.g blue
        orientation_dict[t] = split_key[2] # e.g south
        position_dict[t] = player_sprite.get(single_key)
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

done = False
t=0
window_size = 50
while not done:
    # Example: random actions for each agent
    actions = np.random.randint(action_min,action_max+1, num_players)
    # Step through the environment
    timestep = env.step(actions)

    # Get world RGB from 0th agent (assuming full observability)
    screen_frame = timestep.observation[0]['WORLD.RGB']

    # Get dictionary of states from world RGB frame
    processed = converter.image_to_state(screen_frame)['global']

    # plt.imshow(screen_frame)
    # plt.show()

    # total system reward is social welfare
    processed["social_welfare"] = [sum(timestep.reward).item()] # turn into list for iteration in get_dynamic info

    #print(utils.proc_funcs.get_dynamic_info(processed))
    timestep_dictionary[f"Timestep {t+1}"] = processed

    # Check if episode is done
    done = timestep.last()
    t+=1

    if t==window_size:

        break


result = utils.proc_funcs.dynamic_process(timestep_dictionary)
print(f"\n Dynamic Process after {window_size} steps \n")

for i in range(len(result)):
     print(f"t{i+1}:", end="")
     print(result[f"t_{i+1}"])
