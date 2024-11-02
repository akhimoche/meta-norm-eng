from meltingpot import substrate
from matplotlib import pyplot as plt
from env.mp_llm_env import LLMPrepObject
import numpy as np

# Define the substrate (environment) name
env_name = 'coins'  # example environment from Melting Pot
num_players=2
roles = tuple(['default' for _ in range(num_players)])
# Build the environment using the substrate's configuration
env = substrate.build(env_name, roles=roles)
converter = LLMPrepObject('/home/akhimoche/meta-ssd/sprite_labels/coins')

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.maximum
action_min = action_spec.minimum


obs = env.reset()
done = False
t=0
while not done:
    # Example: random actions for each agent
    actions = np.random.randint(action_min,action_max+1, num_players)
    # Step through the environment
    timestep = env.step(actions)
    #plt.imshow(timestep.observation[0]['WORLD.RGB'], interpolation='nearest')
    #plt.show()
    processed = converter.image_to_state(timestep.observation[0]['WORLD.RGB'])
    if t==150:
        last = timestep.observation

    # Check if episode is done
    done = timestep.last()
    t+=1
# the observation returns a list of agent dictionary observations
agent_0 = last[0]
agent_1 = last[1]

screen_frame = agent_0['WORLD.RGB'] # get world RGB frame
# plt.imshow(screen_frame, interpolation='nearest')
# plt.show()
# Get image print

print(processed)
plt.imshow(screen_frame, interpolation='nearest')
plt.show()