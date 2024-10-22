import meltingpot
from meltingpot import substrate
from dm_env import specs
import numpy as np

# Define the substrate (environment) name
env_name = 'prisoners_dilemma_in_the_matrix__repeated'  # example environment from Melting Pot
num_players=2
roles = tuple(['default' for _ in range(num_players)])
# Build the environment using the substrate's configuration
env = substrate.build(env_name, roles=roles)

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.minimum
action_min = action_spec.maximum
print(num_actions)
print(action_max)
print(action_min)


obs = env.reset()
done = False
while not done:
    # Example: random actions for each agent
    actions = np.random.randint(action_min,action_max+1, num_players)
    # Step through the environment
    timestep = env.step(actions)
    # Check if episode is done
    done = timestep.last()

