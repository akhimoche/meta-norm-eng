# Section 0: Action Space Definition
# Standard action mapping for MeltingPot agents
ACTION_MAP = {
    "NOOP": 0,
    "FORWARD": 1, "BACKWARD": 2,
    "STEP_LEFT": 3, "STEP_RIGHT": 4,
    "TURN_LEFT": 5, "TURN_RIGHT": 6,
    "FIRE_ZAP": 7,
}

# Section 1: Base Agent Class
# Not sure if we need both colour and self id? (to locate self) 
class BaseAgent:
    def __init__(self, agent_id, colour, converter): # Store the agent's id, colour, and vision system
        self.id = agent_id 
        self.colour = colour  # Agent's colour for self-location in symbolic state
        self.converter = converter  # LLMPrepObject for RGB -> symbolic state conversion 

    def act(self, obs): # Demand that teh subclasses implement act() method
        raise NotImplementedError("Subclasses must implement act() method")


# Design Question: Agent Vision 
# VISION SYSTEM UNDERSTANDING:
# - LLMPrepObject in env/mp_llm_env.py handles RGB -> symbolic conversion
# - run_agents.py creates converter and does calibration (finding agent colors, rotating to north)
# - BaseAgent stores converter and provides vision to all agent types
# - Agents call self.converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
# - !! Might want to chaneg to partial observability? Just the RGB (88 x 88), not global.RGB (144 x 192)
# - QUESTION: Is this the best compartmentalization? Should vision be in BaseAgent or elsewhere?