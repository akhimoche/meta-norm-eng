"""
Pulsing Norm for Commons Harvest
--------------------------------
This norm implements a global temporal synchronization strategy to maximize 
social welfare. It alternates between a 'Recovery' phase (all harvesting blocked) 
and a 'Harvest' phase (all harvesting allowed).

Mechanism:
- The norm defines a total cycle length (recovery_duration + harvest_duration).
- If the current timestep is within the 'recovery' window, ALL apples are blocked.
- If the current timestep is within the 'harvest' window, NO apples are blocked.

Why this works:
1. Max Regrowth: By stopping all consumption for 50 steps, we give the probabilistic
   regrowth mechanics the maximum opportunity to refill the board.
2. Coordination: It solves the prisoner's dilemma by enforcing a mandatory 
   cooperation period.
"""

from utils.norms.norm import Norm, Coord
from typing import Set

class PulsingNorm(Norm):
    def __init__(self, name: str = "pulsing_norm", epsilon: float = 0.0, 
                 recovery_duration: int = 50, harvest_duration: int = 10):
        """
        Initializes the Pulsing Norm.

        Args:
            name (str): The name of the norm.
            epsilon (float): Probability (0.0 to 1.0) that an agent ignores the norm.
            recovery_duration (int): Number of steps all apples are blocked.
            harvest_duration (int): Number of steps apples are available.
        """
        super().__init__(name, epsilon)
        self.recovery_duration = recovery_duration
        self.harvest_duration = harvest_duration
        self.cycle_length = recovery_duration + harvest_duration
        
        self.all_apples: Set[Coord] = set()
        self._initialize_apple_locations()

    def _initialize_apple_locations(self):
        """
        Parses the map to identify all apple locations. These are the positions
        that will be toggled on/off during the pulsing cycle.
        """
        ascii_map = [
            "WWWWWWWWWWWWWWWWWWWWWWWW",
            "WAAA    A      A    AAAW",
            "WAA    AAA    AAA    AAW",
            "WA    AAAAA  AAAAA    AW",
            "W      AAA    AAA      W",
            "W       A      A       W",
            "W  A                A  W",
            "W AAA  Q        Q  AAA W",
            "WAAAAA            AAAAAW",
            "W AAA              AAA W",
            "W  A                A  W",
            "W                      W",
            "W                      W",
            "W                      W",
            "W  PPPPPPPPPPPPPPPPPP  W",
            "W PPPPPPPPPPPPPPPPPPPP W",
            "WPPPPPPPPPPPPPPPPPPPPPPW",
            "WWWWWWWWWWWWWWWWWWWWWWWW"
        ]

        for y, row_str in enumerate(ascii_map):
            for x, char in enumerate(row_str):
                if char == 'A':
                    self.all_apples.add((x, y))

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns the blocked positions for timestep t.
        
        Logic:
        - Calculate position in the cycle: t % cycle_length
        - If position < recovery_duration: Block ALL apples.
        - Else (harvest phase): Block NOTHING.
        
        Args:
            t (int): Current timestep.
            
        Returns:
            Set[Coord]: The set of blocked apple coordinates.
        """
        cycle_position = t % self.cycle_length
        
        if cycle_position < self.recovery_duration:
            # RECOVERY PHASE: Protect everything
            return self.all_apples
        else:
            # HARVEST PHASE: Free for all
            return set()